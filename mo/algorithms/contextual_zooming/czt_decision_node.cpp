#include "mo/algorithms/contextual_zooming/czt_decision_node.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <cmath>
#include <limits>
#include <sstream>

using namespace std; 

namespace thts {
    
    CztDNode::CztDNode(
        shared_ptr<CztManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CztCNode> parent,
        bool eval_mo_heuristic) :
            BlThtsDNode(
                static_pointer_cast<BlThtsManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const BlThtsCNode>(parent),
                eval_mo_heuristic),
            _action_ctx_key(),
            _ball_ctx_key()
    {
        stringstream ss_a;
        ss_a << "a_" << decision_depth;
        _action_ctx_key = ss_a.str();
        stringstream ss_b;
        ss_b << "b_" << decision_depth;
        _ball_ctx_key = ss_b.str();
    }
    
    void CztDNode::visit(MoThtsContext& ctx) 
    {
        BlThtsDNode::visit(ctx);
        // num_visits += 1;
    } 

    /**
     * N term for CZT (estimate for total number of trials (that will be run in total))
     * Either the number of visits to this node, or, min_k 2^k s.t. num_visits < 2^k
     */
    double CztDNode::get_N_term(MoThtsContext& ctx) const
    {
        CztManager& manager = (CztManager&) *thts_manager;
        int num_visits = get_num_visits(ctx);

        if (!manager.use_doubling_N_term) {
            return static_cast<double>(num_visits);
        }

        int k = (num_visits > 0) ? static_cast<int>(ceil(log2(num_visits + 1))) : 0;
        k = max(k, manager.min_log2_N);
        return static_cast<double>(1 << k);  // 2^k
    }

    void CztDNode::fill_cz_values_and_ball_ptrs(
        ActionVector& actions,
        unordered_map<shared_ptr<const Action>,double>& cz_values, 
        unordered_map<shared_ptr<const Action>,shared_ptr<CzBall>>& cz_balls, 
        MoThtsContext& ctx) 
    {        
        CztManager& manager = (CztManager&) *thts_manager;
        double opp_coeff = is_opponent() ? -1.0 : 1.0;

        // Compute cz values
        for (shared_ptr<const Action> action : actions) {
            double action_cz_value = std::numeric_limits<double>::lowest();

            // If no child then no ball list
            // Compute the confidence interval of a ball with radius 1 and no visits
            if (!has_child_node_itfc(action)) {
                double unit_ball_radius = 1.0;
                action_cz_value = 2.0 * unit_ball_radius + manager.bias * sqrt(log(get_N_term(ctx)+3));
                cz_values[action] = action_cz_value;
                cz_balls[action] = nullptr;

            } else { //has_child_node(action)
                // "Pre index"
                unordered_map<shared_ptr<CzBall>,double> cz_pre_indices;
                CztCNode& child = *get_child_node(action);
                shared_ptr<vector<shared_ptr<CzBall>>> relevant_balls = child.ball_list.get_relevant_balls(
                    ctx.context_weight.vec); 

                for (shared_ptr<CzBall> ball_ptr : *relevant_balls) {
                    double ball_cz_pre_index_value = 0.0;
                    CzBall& ball = *ball_ptr;

                    ball_cz_pre_index_value += opp_coeff * ball.get_scalarised_avg_return_or_value(ctx.context_weight.vec);
                    ball_cz_pre_index_value += 2.0 * ball.radius();
                    ball_cz_pre_index_value += manager.bias * ball.confidence_radius(get_N_term(ctx));
                    cz_pre_indices[ball_ptr] = ball_cz_pre_index_value;
                }

                // CZ value of action = max of "index values" of all balls
                double action_cz_value = std::numeric_limits<double>::lowest();
                shared_ptr<CzBall> action_cz_ball = nullptr;
                for (shared_ptr<CzBall> ball_ptr : *relevant_balls) {
                    CzBall& ball = *ball_ptr;
                    shared_ptr<vector<shared_ptr<CzBall>>> larger_balls = child.ball_list.get_balls_with_min_radius(
                        ball.radius());

                    double ball_cz_index_value = cz_pre_indices[ball_ptr];
                    for (shared_ptr<CzBall> larger_ball_ptr : *larger_balls) {
                        CzBall& larger_ball = *larger_ball_ptr;
                        double larger_ball_cz_pre_index_value = cz_pre_indices[larger_ball_ptr];
                        double ball_distance = thts::helper::dist(ball.center(), larger_ball.center());
                        double poss_ball_cz_index_value = larger_ball_cz_pre_index_value + ball_distance;
                        if (poss_ball_cz_index_value > ball_cz_index_value) {
                            ball_cz_index_value = poss_ball_cz_index_value;
                        }
                    }

                    if (ball_cz_index_value > action_cz_value) {
                        action_cz_value = ball_cz_index_value;
                        action_cz_ball = ball_ptr;
                    }
                }

                // Add action cz value to map
                cz_values[action] = action_cz_value;
                cz_balls[action] = action_cz_ball;
            }
        }
    }

    shared_ptr<const Action> CztDNode::select_action(MoThtsContext& ctx) 
    {
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,double> cz_values;
        unordered_map<shared_ptr<const Action>,shared_ptr<CzBall>> cz_balls;
        fill_cz_values_and_ball_ptrs(*actions, cz_values, cz_balls, ctx);
        shared_ptr<const Action> result_action = helper::get_max_key_break_ties_randomly(cz_values, *thts_manager);

        // Put action and ball in context
        ctx.put_value_const<Action>(_action_ctx_key, result_action);
        ctx.put_value<CzBall>(_ball_ctx_key, cz_balls[result_action]);

        // Remember to create the child node if it doesnt exist!
        if (!has_child_node_itfc(result_action)) {
            create_child_node(result_action);
        }
        return result_action;
    }

    shared_ptr<const Action> CztDNode::recommend_action(MoThtsContext& ctx) const 
    { 
        shared_ptr<ActionVector> actions = thts_manager->thts_env()->get_valid_actions_itfc(state, ctx);
        unordered_map<shared_ptr<const Action>,double> scalarised_values;

        for (shared_ptr<const Action> action : *actions) {
            scalarised_values[action] = numeric_limits<double>::lowest();
            if (has_child_node_itfc(action)) {
                shared_ptr<vector<shared_ptr<CzBall>>> relevant_balls = 
                    get_child_node(action)->ball_list.get_relevant_balls(ctx.context_weight.vec);
                for (shared_ptr<CzBall> ball_ptr : *relevant_balls) {
                    double ball_val = ball_ptr->get_scalarised_avg_return_or_value(ctx.context_weight.vec);
                    if (ball_val > scalarised_values[action]) {
                        scalarised_values[action] = ball_val;
                    }
                }
            }
        }

        return helper::get_max_key_break_ties_randomly(scalarised_values, *thts_manager);
    }

    void CztDNode::backup(
        const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
        const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
        const Eigen::ArrayXd trial_cumulative_return_after_node, 
        const Eigen::ArrayXd trial_cumulative_return,
        MoThtsContext& ctx) 
    {
        increment_and_update_backup_count();

        shared_ptr<const Action> chosen_action = ctx.get_value_ptr_const<Action>(_action_ctx_key);
        shared_ptr<CzBall> chosen_ball = ctx.get_value_ptr<CzBall>(_ball_ctx_key);
        CztCNode& child = *get_child_node(chosen_action);
        if (chosen_ball == nullptr) {
            // if nullptr, means we made the child node this trial. Get initial ball made in new ball list in new child
            chosen_ball = child.ball_list.get_init_ball();
        }
        child.ball_list.avg_return_update_ball_list(
            trial_cumulative_return_after_node, ctx.context_weight.vec, chosen_ball);
    }

    string CztDNode::get_pretty_print_val() const {
        return "";
    }

    ConvexHull CztDNode::get_convex_hull() const {
        ConvexHull approx_convex_hull;
        for (pair<shared_ptr<const Action>,shared_ptr<ThtsCNode>> pair : children) {
            CztCNode& child = static_cast<CztCNode&>(*pair.second);
            approx_convex_hull |= child.ball_list.get_approximate_convex_hull();
        }
        return approx_convex_hull;
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<CztCNode> CztDNode::create_child_node(shared_ptr<const Action> action) 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<CztCNode>(new_child);
    }

    shared_ptr<BlThtsCNode> CztDNode::create_child_node_helper(shared_ptr<const Action> action) const 
    {   
        shared_ptr<CztCNode> new_child = make_shared<CztCNode>(
            static_pointer_cast<CztManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const CztDNode>(shared_from_this()));
        return static_pointer_cast<BlThtsCNode>(new_child);
    }

    shared_ptr<CztCNode> CztDNode::get_child_node(shared_ptr<const Action> action) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<CztCNode>(new_child);
    }
}