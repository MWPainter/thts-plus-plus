#include "algorithms/ments/dents/dents_chance_node.h"

using namespace std;

namespace thts {
    /**
     * Constructor
     */
    DentsCNode::DentsCNode(
        shared_ptr<DentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DentsDNode> parent) :
            DBMentsCNode(
                static_pointer_cast<MentsManager>(thts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsDNode>(parent)),
            EntCNode(),
            EmpNode()
    {
    }

    /**
     * Get decayed entropy coeff
     */
    double DentsCNode::get_entropy_coeff() const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        Schedule& entropy_coeff_schedule = *manager.entropy_coeff_schedule_ptr;
        return entropy_coeff_schedule(num_visits);
    }

    /**
     * Calls entropy backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void DentsCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsContext& ctx)
    {   
        MentsCNode::num_backups++;

        // entropy backup
        backup_ent<DentsDNode>(children, empirical_distribution);

        // value backup
        double val_estimate;
        DentsManager& manager = (DentsManager&) *thts_manager;
        if (manager.use_dp_value) {
            backup_dp<DentsDNode>(children, empirical_distribution, local_reward, is_opponent());
            val_estimate = dp_value;
        } else {
            backup_emp(trial_cumulative_return_after_node);
            val_estimate = avg_return;
        }
    
        // update local soft_value so that value is sensible / for pretty printing
        soft_value = val_estimate + get_entropy_coeff() * subtree_entropy;
    }

    /**
     * Make child node
     */
    shared_ptr<DentsDNode> DentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<DentsDNode>(
            static_pointer_cast<DentsManager>(ThtsCNode::thts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const DentsCNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string DentsCNode::get_pretty_print_val() const {
        DentsManager& manager = (DentsManager&) *thts_manager;
        double val_estimate = manager.use_dp_value ? dp_value : avg_return; 

        stringstream ss;
        ss << val_estimate << "(entrpy:" << subtree_entropy << ",val_temp:" << get_entropy_coeff() << ",soft_val:" 
            << soft_value << ")";
        return ss.str();
    }
}

/**
 * Boilerplate ThtsCNode interface implementation. Copied from thts_chance_node_template.h.
 */
namespace thts {
    shared_ptr<ThtsDNode> DentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<DentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}
