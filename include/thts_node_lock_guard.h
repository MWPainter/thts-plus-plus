#pragma once

#include <vector>

#include "thts_node.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"

namespace thts
{
    /**
     * Class to perform safe locking around a ThtsNode for coarse locking. If a node is locked using ThtsNodeLockGuard, 
     * then the thread has acquired locks for the node and all of its children, which is sufficient to perform 
     * selection and backups.
     * 
     * As we need to account for nodes potentially being in a graph, rather than a tree, we have to lock in a strict 
     * ordering. (In a tree the strict ordering comes from the parent/child relationship).
     * 
     * As such, this class represents the logic to aquire the lock for a ThtsNode, and all of its children, in a 
     * thread safe way that avoids deadlock.
     * 
     * Additionally, it is worth noting that we need to keep track of what nodes we locked. We cannot just lock a node 
     * and all children, and then unlock the node and all its children. This is because during the selection phase, we 
     * may add children to a node, and if that node came from the transposition table, then this thread doesn't have 
     * the lock for that child. So we must keep track of what nodes we locked.
     * 
     * v1TODO: should add some unit test for the case just discribed.
     * 
     * v1TODO: docstring about RAII or whatever that's called.
     * 
     * v1TODO: should move children map to ThtsNode definition, then dont need 2 constructors that are identical
     */
    class ThtsNodeLockGuard {

        protected:
            std::vector<std::shared_ptr<ThtsNode>> locked_nodes;

        public:
            ThtsNodeLockGuard(std::shared_ptr<ThtsDNode> node);
            ThtsNodeLockGuard(std::shared_ptr<ThtsCNode> node);

            virtual ~ThtsNodeLockGuard();
    };
}