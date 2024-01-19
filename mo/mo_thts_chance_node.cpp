#include "mo/mo_thts_chance_node.h"

using namespace std;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    MoThtsCNode::MoThtsCNode(
        shared_ptr<MoThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MoThtsDNode> parent) :
            ThtsCNode(thts_manager, state, action, decision_depth, decision_timestep, parent)
    {
    }

    /**
     * Raise error if call wrong backup fn
    */
    void MoThtsCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        throw runtime_error("Called single objective backup function for multi objective node");
    }
}