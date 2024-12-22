#include "mo/smt_dents_decision_node.h"

#include "helper_templates.h"
#include "algorithms/common/decaying_temp.h"
#include "mo/mo_helper.h"

#include <limits>
#include <sstream>

using namespace std; 

namespace thts {
    SmtDentsDNode::SmtDentsDNode(
        shared_ptr<SmtDentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const SmtDentsCNode> parent) :
            SmtBtsDNode(
                static_pointer_cast<SmtBtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const SmtBtsCNode>(parent))
    {
    }

    double SmtDentsDNode::get_value_temp() const {
        SmtDentsManager& manager = (SmtDentsManager&) *thts_manager;
        return compute_decayed_temp(
            manager.value_temp_decay_fn, 
            manager.value_temp_init, 
            manager.value_temp_decay_min_temp, 
            num_visits, 
            manager.value_temp_decay_visits_scale);
    }

    void SmtDentsDNode::compute_action_weights(
        ActionVector& actions,
        unordered_map<shared_ptr<const Action>,Eigen::ArrayXd>& q_val_map,
        unordered_map<shared_ptr<const Action>,double>& entropy_map, 
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        MoThtsContext& context) const
    {
        // TODO: cleaner use of opp coeff... and better documentation of what things mean
        // E.g. get_q_value should return without opp coeff
        // Then only use opp coeff locally where necessary
        // Also make it "get_opp_coeff" as an inline version of this
        // tHIS is an across library thing to do
        double opp_coeff = is_opponent() ? -1.0 : 1.0;

        // get temp
        double temp = get_temp();
        double val_temp = get_value_temp();

        // compute normalisation term (ctx_val already includes opp coeff)
        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : actions) {
            double ctx_val = thts::helper::dot(context.context_weight, q_val_map[action]);
            ctx_val += opp_coeff * val_temp * entropy_map[action];
            double ctx_val_over_temp = ctx_val / temp;
            if (normalisation_term < ctx_val_over_temp) {
                normalisation_term = ctx_val_over_temp;
            }
        }

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : actions) {
            double ctx_q_value = thts::helper::dot(context.context_weight, q_val_map[action]);
            ctx_q_value += opp_coeff * val_temp * entropy_map[action];
            double action_weight = exp((ctx_q_value/temp) - normalisation_term);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    /**
     * See comments on NGV datatype for what the pure_backup stuff is about
     */
    void SmtDentsDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        SmtDentsManager& manager = (SmtDentsManager&) *thts_manager;
        num_backups++;

        // Get closest NGV in simplex map
        shared_ptr<TN> simplex = simplex_map.get_leaf_tn_node(ctx.context_weight);
        shared_ptr<NGV> closest_vertex = simplex->get_closest_ngv_vertex(ctx.context_weight);

        // Make list of vertices to backup
        vector<shared_ptr<NGV>> vertices_to_backup;
        vertices_to_backup.push_back(closest_vertex);
        if (manager.backup_all_vertices_of_simplex) {
            vector<shared_ptr<NGV>>& simplex_vertices = *simplex->simplex_vertices;
            vertices_to_backup.insert(vertices_to_backup.end(), simplex_vertices.begin(), simplex_vertices.end());
        }

        // Only get actions once - avoid pointless RPC calls
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);

        // Backup at all vertices
        for (shared_ptr<NGV> ngv : vertices_to_backup) {

            // Want to backup according to ngv->weight, so make a context with that weight to use
            MoThtsContext ngv_ctx(ngv->weight);

            // get q vals
            unordered_map<shared_ptr<const Action>,Eigen::ArrayXd> q_val_map;
            unordered_map<shared_ptr<const Action>,double> entropy_map;
            unordered_map<shared_ptr<const Action>,bool> pure_backup_val_map;
            get_child_q_values(*actions, q_val_map, entropy_map, pure_backup_val_map, ngv_ctx);

            // Compute local policy + entropy
            ActionDistr policy;
            compute_action_distribution(*actions, q_val_map, entropy_map, policy, ngv_ctx);

            double local_entropy = 0.0;
            for (pair<shared_ptr<const Action>,double> pr : policy) {
                double prob = pr.second;
                if (prob == 0.0) continue;
                local_entropy -= prob * log(prob); 
            }

            // dp backups + subtree entropy
            Eigen::ArrayXd best_q_val;
            double subtree_entropy = 0.0;
            bool best_q_val_is_pure_backup = false;
            double max_ctx_q_val = numeric_limits<double>::lowest();
            for (shared_ptr<const Action> action : *actions) {
                double ctx_q_val = thts::helper::dot(ngv->weight, q_val_map[action]);
                if (ctx_q_val > max_ctx_q_val) {
                    max_ctx_q_val = ctx_q_val;
                    best_q_val = q_val_map[action];
                    best_q_val_is_pure_backup = pure_backup_val_map[action];
                }
                subtree_entropy += policy[action] * entropy_map[action];
            }

            // update simplex map
            double opp_coeff = is_opponent() ? -1.0 : 1.0;
            ngv->value_estimate = best_q_val * opp_coeff;
            ngv->entropy = local_entropy + subtree_entropy;
            ngv->pure_backup_value_estimate = best_q_val_is_pure_backup;
        }

        // Message passing / sharing + splitting
        for (shared_ptr<NGV> ngv : vertices_to_backup) {
            SmtBtsManager& manager = (SmtBtsManager&) *thts_manager;
            simplex->maybe_subdivide(manager);
            closest_vertex->share_values_message_passing();
        }
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<SmtThtsCNode> SmtDentsDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<SmtDentsCNode> new_child = make_shared<SmtDentsCNode>(
            static_pointer_cast<SmtDentsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const SmtDentsDNode>(shared_from_this()));
        return static_pointer_cast<SmtThtsCNode>(new_child);
    }
}