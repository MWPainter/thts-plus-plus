#include "algorithms/uct/alphago_decision_node.h"

using namespace std; 

namespace thts {
    /**
     * Construct Puct Decision node.
     */
    AlphaGoDNode::AlphaGoDNode(
        shared_ptr<AlphaGoManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const AlphaGoCNode> parent) :
            PuctDNode(
                static_pointer_cast<PuctManager>(thts_manager), 
                state, 
                decision_depth, 
                decision_timestep, 
                static_pointer_cast<const PuctCNode>(parent)),
            base_policy_prior(make_shared<ActionPrior>(*policy_prior))
    {
    }

    /**
     * Add dirichlet noise
     */
    void AlphaGoDNode::add_dirichlet_noise_to_prior() {
        AlphaGoManager& manager = (AlphaGoManager&) *thts_manager;
        if (manager.dirichlet_noise_coeff == 0.0) {
            return;
        }
        vector<double> eta = manager.sample_dirichlet(base_policy_prior->size());
        int i = 0;
        for (pair<shared_ptr<const Action>,double> pr : *base_policy_prior) {
            shared_ptr<const Action> action = pr.first;
            double weight = pr.second;
            (*policy_prior)[action] = (1.0 - manager.dirichlet_noise_coeff) * weight + manager.dirichlet_noise_coeff * eta[i++];
        }
    }

    /**
     * Visit just calls parent version, but adds dirichlet noise to prior if root node
    */
    void AlphaGoDNode::visit(ThtsEnvContext& ctx) {
        PuctDNode::visit(ctx);
        if (is_root_node()) {
            add_dirichlet_noise_to_prior();
        }
    }
    
    /**
     * Make a new PuctCNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<AlphaGoCNode> AlphaGoDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<AlphaGoCNode>(
            static_pointer_cast<AlphaGoManager>(thts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const AlphaGoDNode>(shared_from_this()));
    }

    /**
     * Make create_child functions make a Puct child rather than a UctCNode or ThtsCNode
     */
    shared_ptr<ThtsCNode> AlphaGoDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<AlphaGoCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<ThtsCNode>(child_node);
    }

    /**
     * Point visit itfc at our visit function
    */
    void AlphaGoDNode::visit_itfc(ThtsEnvContext& ctx) {
        visit(ctx);
    }
}