#include "algorithms/uct/hmcts_decision_node.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <float.h>
#include <sstream>
#include <vector>

using namespace std; 

namespace thts {
    /**
     * Constructor, inits members. 
     * 
     * If we have a heuristic function, then initialises 'num_visits' and 'avg_return' according to the heuristic in 
     * the manager. If we have a prior function, then
     */
    HmctsDNode::HmctsDNode(
        shared_ptr<HmctsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const HmctsCNode> parent) :
            UctDNode(
                static_pointer_cast<UctManager>(thts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const UctCNode>(parent)),
            total_budget(0),
            total_budget_on_last_visit(0),
            seq_halving_round_budget_per_child(0),
            seq_halving_actions()
    {   
    }

    /**
     * Running seq halving at this node?
    */
    bool HmctsDNode::running_seq_halving() const {
        HmctsManager& manager = (HmctsManager&) *thts_manager;
        return total_budget > manager.uct_budget_threshold;
    }

    /**
     * Setter
    */
    void HmctsDNode::set_new_total_budget(int budget) {
        total_budget = budget;
    }

    /**
     * Visit for seq halfing mode
     * 
     * Zeroth (forgot this until was implementing run_toy bit). If root node, no one is going to set our budget, so 
     * do it ourselves
     * 
     * Firstly, update total_budget_on_last_visit if necessary. If it was out of date, that means that our budget was 
     * updated, and we should reset our sequential halving routines. N.B. the copy constructor of std::vector will 
     * copy the vector to be independent
     * 
     * Secondly, if size(children) != size(actions), then this is likely our first visit to this node, so no children
     * and things are setup correctly from here
     * 
     * Thirdly, if all children have been created, checks if we need to start the next round of sequential halving, by 
     * checking if all children have used their computational budget. I think its very possible (if we got a small 
     * budget increase near the end of running HMCTS) that the children nodes may have been visited enough already for 
     * the first few rounds. So we update what round of sequential halving we are on until at least one node has 
     * outstanding budget, or until there is only one action. 
     * 
     * Fourthly, update the total budget for child nodes
     * 
     * Do all of 3 and 4 under locking to preserve threading serialisation with minimal headaches
    */
    void HmctsDNode::visit_update_budgets() {
        if (is_root_node() && num_visits == 0) {
            HmctsManager& manager = (HmctsManager&) *thts_manager;
            total_budget = manager.total_budget;
        }

        if (total_budget != total_budget_on_last_visit) {
            total_budget_on_last_visit = total_budget;
            ActionVector seq_halving_actions = *actions;
            seq_halving_round_budget_per_child = floor(
                ((double) num_visits + total_budget) / (seq_halving_actions.size() * ceil(log2(actions->size()))) );
            if (seq_halving_round_budget_per_child < 1) {
                seq_halving_round_budget_per_child = 1;
            }
        }

        if (children.size() != actions->size()) {
            return;
        }

        lock_all_children();
        while (seq_halving_actions.size() > 1) {
            // check for outstanding budget
            bool child_has_outstanding_budget = false;
            for (shared_ptr<const Action> act : seq_halving_actions) {
                HmctsCNode& child = (HmctsCNode&) *get_child_node(act);
                if (child.num_visits < seq_halving_round_budget_per_child) {
                    child_has_outstanding_budget = true;
                    break;
                }
            }
            if (child_has_outstanding_budget) {
                break;
            }

            // make new seq_halving_actions array, by sorting and taking the first half
            int new_num_actions = ceil(seq_halving_actions.size() / 2.0);
            std::sort(
                seq_halving_actions.begin(), 
                seq_halving_actions.end(), 
                [&](shared_ptr<const Action> a, shared_ptr<const Action> b) {
                    return get_child_node(a)->avg_return > get_child_node(b)->avg_return;
                });
            ActionVector::const_iterator first = seq_halving_actions.begin();
            ActionVector::const_iterator last = seq_halving_actions.begin() + new_num_actions;
            seq_halving_actions = ActionVector(first, last);

            // Update budget per child
            int additional_budget = floor(
                ((double) num_visits + total_budget) / (new_num_actions + ceil(log2(actions->size()))) );
            if (additional_budget < 1) {
                additional_budget = 1;
            }
            seq_halving_round_budget_per_child += additional_budget;
        }

        for (shared_ptr<const Action> act : seq_halving_actions) {
            get_child_node(act)->set_new_total_budget(seq_halving_round_budget_per_child);
        }
        unlock_all_children();
    }

    /**
     * Parent node will set the total budget. 
     * Update local seq halving variables if needed
     */
    void HmctsDNode::visit(ThtsEnvContext& ctx) {
        if (running_seq_halving()) {
            visit_update_budgets();
        }
        UctDNode::visit_itfc(ctx);
    }

    /**
     * Selects action from actions with the most budget left randomly
     * 
     * First checks that all children exist though, and pulls an uninitialised arm if not
     * 
     * Realised when writing this that we might accidentally double select a child when using multithreading leading 
     * to one being picked slightly more than the other. But with the randomisation it should be ok enough
     */
    shared_ptr<const Action> HmctsDNode::select_action_sequential_halving(ThtsEnvContext& ctx) {
        // Pull uninitialised arms if needed
        vector<shared_ptr<const Action>> actions_yet_to_try;
        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) {
                actions_yet_to_try.push_back(action);
            }
        }

        if (actions_yet_to_try.size() > 0) {
            int indx = thts_manager->get_rand_int(0,actions_yet_to_try.size());
            shared_ptr<const Action> action = actions_yet_to_try[indx];
            create_child_node(action);
            return action;
        }

        vector<shared_ptr<const Action>> max_budget_remaining_actions;
        int max_budget_remaining = -1;
        for (shared_ptr<const Action> act : seq_halving_actions) {
            HmctsCNode& child = (HmctsCNode&) *get_child_node(act);
            int child_remaining_budget = seq_halving_round_budget_per_child - child.num_visits;
            if (child_remaining_budget > max_budget_remaining) {
                max_budget_remaining_actions = { act };
                max_budget_remaining = child_remaining_budget;
            } else if (child_remaining_budget == max_budget_remaining) {
                max_budget_remaining_actions.push_back(act);
            }
        }

        int indx = thts_manager->get_rand_int(0, max_budget_remaining_actions.size());
        return max_budget_remaining_actions[indx];
    }
    
    /**
     * Select action function.
     * 
     * If doing sequential halving here, then use that function, otherwise uct
     */
    shared_ptr<const Action> HmctsDNode::select_action(ThtsEnvContext& ctx) {
        if (running_seq_halving()) {
            return select_action_sequential_halving(ctx);
        }
        return UctDNode::select_action(ctx);
    }
    
    /**
     * Make a child
     */
    shared_ptr<HmctsCNode> HmctsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        shared_ptr<HmctsCNode> new_child = make_shared<HmctsCNode>(
            static_pointer_cast<HmctsManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const HmctsDNode>(shared_from_this()));
        new_child->set_new_total_budget(seq_halving_round_budget_per_child);
        return new_child;
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace thts {
    shared_ptr<HmctsCNode> HmctsDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<ThtsCNode> new_child = ThtsDNode::create_child_node_itfc(action);
        return static_pointer_cast<HmctsCNode>(new_child);
    }

    shared_ptr<HmctsCNode> HmctsDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ThtsCNode> new_child = ThtsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<HmctsCNode>(new_child);
    }
}

/**
 * Boilerplate ThtsDNode interface implementation. Copied from thts_decision_node_template.h.
 */
namespace thts {
    void HmctsDNode::visit_itfc(ThtsEnvContext& ctx) {
        visit(ctx);
    }

    shared_ptr<const Action> HmctsDNode::select_action_itfc(ThtsEnvContext& ctx) {
        return select_action(ctx);
    }

    shared_ptr<const Action> HmctsDNode::recommend_action_itfc(ThtsEnvContext& ctx) const {
        return recommend_action(ctx);
    }

    void HmctsDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx);
    }

    shared_ptr<ThtsCNode> HmctsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<HmctsCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<ThtsCNode>(child_node);
    }
}