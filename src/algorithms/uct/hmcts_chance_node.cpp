#include "algorithms/uct/hmcts_chance_node.h"

#include "helper_templates.h"

using namespace std; 

namespace thts {
    /**
     * Construct Hmcts Chance node. Use thts_manager to initialise values with heuristic if necessary.
     */
    HmctsCNode::HmctsCNode(
        shared_ptr<HmctsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const HmctsDNode> parent) :
            UctCNode(
                static_pointer_cast<UctManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const UctDNode>(parent)),
            total_budget(0),
            total_budget_on_last_visit(0),
            budget_per_child()
    {  
    }

    /**
     * Running seq halving at this node?
    */
    bool HmctsCNode::running_seq_halving() const {
        HmctsManager& manager = (HmctsManager&) *thts_manager;
        return total_budget > manager.uct_budget_threshold;
    }

    /**
     * Setter
    */
    void HmctsCNode::set_new_total_budget(int budget) {
        total_budget = budget;
    }

    /**
     * Visit for seq halfing mode
     * 
     * Just distributes the budget proportionally
    */
    void HmctsCNode::visit_update_budgets() {
        if (total_budget != total_budget_on_last_visit) {
            total_budget_on_last_visit = total_budget;
            budget_per_child.reserve(next_state_distr->size());
            for (pair<shared_ptr<const State>,double> pr : *next_state_distr) {
                shared_ptr<const State> state = pr.first;
                double prob = pr.second;
                budget_per_child[state] = ceil(prob * total_budget);
            }

            lock_all_children();
            for (pair<shared_ptr<const State>,double> pr : *next_state_distr) {
                shared_ptr<const State> state = pr.first;
                if (has_child_node(state)) {
                    HmctsDNode& child = (HmctsDNode&) *get_child_node(state);
                    child.set_new_total_budget(budget_per_child.at(state));
                }
            }
            unlock_all_children();
        }
    }

    /**
     * Parent node will set the total budget. 
     * Update local seq halving variables if needed
     */
    void HmctsCNode::visit(ThtsEnvContext& ctx) {
        if (running_seq_halving()) {
            visit_update_budgets();
        }
        UctCNode::visit_itfc(ctx);
    }

    /**
     * Implementation of sample_observation, that uses the sample from distribution helper function.
     * Normalises the budgets remaining using the probabilities
     */
    shared_ptr<const State> HmctsCNode::sample_observation_budgeted() {
        // Pull uninitialised arms if needed
        vector<shared_ptr<const State>> outcomes_yet_to_try;
        for (pair<shared_ptr<const State>,double> pr : *next_state_distr) {
            shared_ptr<const State> state = pr.first;
            if (!has_child_node(state)) {
                outcomes_yet_to_try.push_back(state);
            }
        }

        if (outcomes_yet_to_try.size() > 0) {
            int indx = thts_manager->get_rand_int(0,outcomes_yet_to_try.size());
            shared_ptr<const State> outcome = outcomes_yet_to_try[indx];
            create_child_node(outcome);
            return outcome;
        }

        vector<shared_ptr<const State>> max_budget_remaining_outcomes;
        double max_budget_remaining = -1;
        for (pair<shared_ptr<const State>,int> pr : budget_per_child) {
            shared_ptr<const State> state = pr.first;
            int child_budget = pr.second;
            double prob = next_state_distr->at(state);
            HmctsDNode& child = (HmctsDNode&) *get_child_node(state);
            double child_remaining_budget = (child_budget - child.num_visits) / prob;
            if (child_remaining_budget > max_budget_remaining) {
                max_budget_remaining_outcomes = { state };
                max_budget_remaining = child_remaining_budget;
            } else if (child_remaining_budget == max_budget_remaining) {
                max_budget_remaining_outcomes.push_back(state);
            }
        }

        int indx = thts_manager->get_rand_int(0, max_budget_remaining_outcomes.size());
        return max_budget_remaining_outcomes[indx];
    }
    
    /**
     * Sample observation calls sample_observation_random.
     * Or the seq halving version if we're running that at the moment
     */
    shared_ptr<const State> HmctsCNode::sample_observation(ThtsEnvContext& ctx) {
        if (running_seq_halving()) {
            return sample_observation_budgeted();
        }
        return sample_observation_random();
    }

    /**
     * Make a new HmctsDNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<HmctsDNode> HmctsCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        shared_ptr<HmctsDNode> new_child = make_shared<HmctsDNode>(
            static_pointer_cast<HmctsManager>(thts_manager), 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const HmctsCNode>(shared_from_this()));
        int budget = 0;
        if (!budget_per_child.empty()) {
            budget_per_child.at(next_state);
        }
        new_child->set_new_total_budget(budget);
        return new_child;
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<HmctsDNode> HmctsCNode::create_child_node(shared_ptr<const State> observation) 
    {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<HmctsDNode>(new_child);
    }
    
    shared_ptr<HmctsDNode> HmctsCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<ThtsDNode> new_child = ThtsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<HmctsDNode>(new_child);
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    void HmctsCNode::visit_itfc(ThtsEnvContext& ctx) {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> HmctsCNode::sample_observation_itfc(ThtsEnvContext& ctx) {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        shared_ptr<const State> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void HmctsCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        ThtsEnvContext& ctx_itfc = (ThtsEnvContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<ThtsDNode> HmctsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<HmctsDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}