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

    void MoThtsCNode::visit_itfc(ThtsEnvContext& ctx) {
        ThtsCNode::visit_itfc(ctx);
        MoThtsContext& mo_ctx = *dynamic_pointer_cast<MoThtsContext>(ctx);
        vector_visit_count += ctx.context_weight;
    }

    double MoThtsCNode::get_num_visits(ThtsEnvContext& ctx) {
        MoThtsManager& mo_thts_manager = *dynamic_pointer_cast<MoThtsManager>(thts_manager());
        MoThtsContext& mo_ctx = *dynamic_pointer_cast<MoThtsContext>(ctx);
        if (mo_thts_manager.use_vector_visit_counts) {
            return vector_visit_count.dot(ctx.context_weight)
        }
        return num_visits;
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