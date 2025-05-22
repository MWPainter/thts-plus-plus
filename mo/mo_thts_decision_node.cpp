#include "mo/mo_thts_decision_node.h"

#include "mo/mo_thts_manager.h"

#include <stdexcept>
#include <tuple>
#include <utility>

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     * 
     * Nuance use of heuristic value is to enforce nodes for sink states to have a value of zero
     */
    MoThtsDNode::MoThtsDNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsCNode> parent) :
            ThtsDNode(thts_manager, state, decision_depth, decision_timestep, parent),
            mo_heuristic_value(thts_manager->reward_dim),
            vector_visit_count(thts_manager->reward_dim)
    {
        if (thts_manager->mo_heuristic_fn != nullptr
            && !thts_manager->thts_env()->is_sink_state_itfc(state, *thts_manager->get_thts_context()))
        {
            mo_heuristic_value = thts_manager->mo_heuristic_fn(state, thts_manager->thts_env());
        }
    }

    void MoThtsDNode::visit_itfc(ThtsEnvContext& ctx) {
        ThtsDNode::visit_itfc(ctx);
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        vector_visit_count += mo_ctx.context_weight;
    }

    double MoThtsDNode::get_num_visits(ThtsEnvContext& ctx) const {
        MoThtsManager& mo_thts_manager = (MoThtsManager&) *thts_manager;
        MoThtsContext& mo_ctx = (MoThtsContext&) ctx;
        if (mo_thts_manager.use_vector_visit_counts) {
            return vector_visit_count.dot(mo_ctx.context_weight);
        }
        return num_visits;
    }

    double MoThtsDNode::get_scalar_num_visits() const {
        return num_visits;
    }

    Vec MoThtsDNode::get_vector_num_visits() const {
        return vector_visit_count;
    }

    /**
     * Raise error if call wrong backup fn
    */
    void MoThtsDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        throw runtime_error("Called single objective backup function for multi objective node");
    }
}