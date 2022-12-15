#pragma once

#include "algorithms/common/dp_chance_node.h"

#include "thts_types.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"


namespace thts {
    // forward declare corresponding DPCNode class
    class DPCNode;

    // Typedef for children map
    typedef std::unordered_map<std::shared_ptr<const Action>, std::shared_ptr<DPCNode>> DPCNodeChildMap;

    /**
     * An implementation of dynamic programming backups for nodes to use.
     * 
     * Member variables:
     *      num_backups: The number of backups this node has performed (== "number of visits" with respect to dp backup)
     *      dp_value: The dynamic programming value at this node
     *      thts_manager: A reference to the ultimate thts manager (reference is ok as node has a pointer anyway)
     */
    class DPDNode {
        // Alloow DPCNode access to private members
        friend DPCNode;

        protected:
            int num_backups;
            double dp_value;
            ThtsManager& thts_manager;

            /**
             * Constructor 
             */
            DPDNode(ThtsManager& thts_manager) : num_backups(0), dp_value(0.0), thts_manager(thts_manager) {};

            /**
             * Destructor
             */
           virtual ~DPDNode() = default;

            /**
             * Visit function. 
             * 
             * Necessry so leaf nodes values of 'num_backups' are reliable
             * 
             * Args:
             *      is_leaf: If this node is a leaf node or not
             */
            void visit_dp(bool is_leaf);

            /**
             * Returns an action recommendation from this node.
             * 
             * Assumes all children are locked.
             * 
             * Args:
             *      children: The children map for this node
             *      visit_threshold: 
             *          A threshold value of visits to recommend a child node (if no child has met the threshold then 
             *          we still recommend the highest value ignoring any thresholding).
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             * 
             * Returns:
             *      An action recommendation from this node.
             */
            std::shared_ptr<const Action> recommend_action_best_dp_value_impl(
                DPCNodeChildMap& children, int visit_threshold, bool is_opponent) const;

            /**
             * Performs a dynamic programming backup.
             * 
             * I.e. V(s) = max_a Q(s,a)
             * 
             * Assumes all children are locked.
             * 
             * Args:
             *      children: The children map for this node
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            void backup_dp_impl(DPCNodeChildMap& children, bool is_opponent);

            /**
             * Helper to convert children maps into children maps for DP Nodes.
             * 
             * Templated with the top-level class of the ThtsCNode, so that the children can be cast from
             * ThtsCNode -> T -> DPCNode
             * 
             * Args:
             *      children: A children map for a ThtsDNode, mapping actions to CNodes.
             * 
             * Returns:
             *      A map from actions to DPCNodes, to be used in recommend_action or backup.
             */
            template <typename T>
            std::shared_ptr<DPCNodeChildMap> convert_child_map(const CNodeChildMap& children) const {
                std::shared_ptr<DPCNodeChildMap> dp_children = std::make_shared<DPCNodeChildMap>();
                for (std::pair<std::shared_ptr<const Action>,std::shared_ptr<ThtsCNode>> pr : children) {
                    std::shared_ptr<T> superclass_ptr = std::static_pointer_cast<T>(pr.second);
                    std::shared_ptr<DPCNode> dp_node_ptr = std::static_pointer_cast<DPCNode>(superclass_ptr);
                    dp_children->insert_or_assign(pr.first, dp_node_ptr);
                }
                return dp_children;
            }

        public:
            /**
             * Interface for calling the recommend_action function for ThtsDNode classes subclassing this DPDNode.
             * 
             * Casts the child map so that the DPDNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the ThtsCNode, so that the children can be cast from
             * ThtsCNode -> T -> DPCNode.
             * 
             * Args:
             *      children: The children map for a ThtsDNode (that are ultimately of type T)
             *      visit_threshold: 
             *          A threshold value of visits to recommend a child node (if no child has met the threshold then 
             *          we still recommend the highest value ignoring any thresholding).
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             * 
             * Returns:
             *      An action recommendation from this node.
             */
            template <typename T>
            std::shared_ptr<const Action> recommend_action_best_dp_value(
                const CNodeChildMap& children, int visit_threshold=0, bool is_opponent=false) const 
            {
                std::shared_ptr<DPCNodeChildMap> dp_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                std::shared_ptr<const Action> action = recommend_action_best_dp_value_impl(
                    *dp_children, visit_threshold, is_opponent);
                for (auto pr : children) pr.second->unlock();
                return action;
            }
            
            /**
             * Interface for calling the backup function for ThtsDNode classes subclassing this DPDNode.
             * 
             * Casts the child map so that the DPDNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the ThtsCNode, so that the children can be cast from
             * ThtsCNode -> T -> DPCNode.
             * 
             * Args:
             *      children: The children map for a ThtsDNode (that are ultimately of type T)
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            template <typename T>
            void backup_dp(const CNodeChildMap& children, bool is_opponent=false) {
                std::shared_ptr<DPCNodeChildMap> dp_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                backup_dp_impl(*dp_children, is_opponent);
                for (auto pr : children) pr.second->unlock();
            }
    };
}