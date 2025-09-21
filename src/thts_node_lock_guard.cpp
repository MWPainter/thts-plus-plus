#include "thts_node_lock_guard.h"

#include <algorithm>
#include <thread>
#include <iostream>

namespace thts 
{
    using namespace std;

    /**
     * each loop:
     *  0 = clears lists of nodes for a fresh attempt to lock
     *  1 = gets list of nodes to lock
     *  2 = sort nodes by pointers to provide a strict ordering over nodes
     *  3 = attempts to lock the node and all children (in that strict ordering, and keeping track of successful locks)
     *  4 = checks that we successfully have locked the node and all children (#locked == #children + 1)
     *      it is worth noting at the start of 4, we may or may not have the lock for node, which is why we lock the 
     *      node's *recursive* mutex around the check
     *      If successful, then we are done and can return from the constructor!
     *  5 = if unsuccessfull, unlock all of the locks we successfully locked, then we failed to get all the locks we need
     *  6 = yield, try to let the thread that's got the locks we need finish it's work so we can do ours
     */
    ThtsNodeLockGuard::ThtsNodeLockGuard(shared_ptr<ThtsDNode> node) :
        locked_nodes()
    {
        vector<shared_ptr<ThtsNode>> nodes_to_lock;

        while (true) {
            // clear vectors
            locked_nodes.clear();
            nodes_to_lock.clear();
            nodes_to_lock.push_back(node);
            
            // get list of nodes to lock
            {
                lock_guard<recursive_mutex> lg(node->lock);
                for (auto& [action, child] : node->children)
                {
                    nodes_to_lock.push_back(child);
                }
            }

            // try to lock them in strict ordering
            std::sort(nodes_to_lock.begin(), nodes_to_lock.end());
            for (shared_ptr<ThtsNode>& node_to_lock : nodes_to_lock) {
                bool success = node_to_lock->lock.try_lock();
                if (!success) 
                {
                    break;
                }
                locked_nodes.push_back(node_to_lock);
            }
            
            // Check if we were successful (and return)
            {
                lock_guard<recursive_mutex> lg(node->lock);
                if (locked_nodes.size() == node->children.size() + 1) 
                {
                    return;
                }
            }

            // we were unsuccssfull, unlock everything we locked
            for (shared_ptr<ThtsNode>& node_to_unlock : locked_nodes)
            {
                node_to_unlock->lock.unlock();
            }             

            // yield
            std::this_thread::yield();
        }
    } 

    /**
     * C&P of ThtsDNode version
     */
    ThtsNodeLockGuard::ThtsNodeLockGuard(shared_ptr<ThtsCNode> node) :
        locked_nodes()
    {
        vector<shared_ptr<ThtsNode>> nodes_to_lock;

        while (true) {
            // clear vectors
            locked_nodes.clear();
            nodes_to_lock.clear();
            nodes_to_lock.push_back(node);
            
            // get list of nodes to lock
            {
                lock_guard<recursive_mutex> lg(node->lock);
                for (auto& [obs, child] : node->children)
                {
                    nodes_to_lock.push_back(child);
                }
            }

            // try to lock them in strict ordering
            std::sort(nodes_to_lock.begin(), nodes_to_lock.end());
            for (shared_ptr<ThtsNode>& node_to_lock : nodes_to_lock) {
                bool success = node_to_lock->lock.try_lock();
                if (!success) 
                {
                    break;
                }
                locked_nodes.push_back(node_to_lock);
            }
            
            // Check if we were successful (and return)
            {
                lock_guard<recursive_mutex> lg(node->lock);
                if (locked_nodes.size() == node->children.size() + 1) 
                {
                    return;
                }
            }

            // we were unsuccssfull, unlock everything we locked
            for (shared_ptr<ThtsNode>& node_to_unlock : locked_nodes)
            {
                node_to_unlock->lock.unlock();
            }             

            // yield
            std::this_thread::yield();
        }
    }

    ThtsNodeLockGuard::~ThtsNodeLockGuard() 
    {
        for (shared_ptr<ThtsNode> node : locked_nodes) 
        {
            node->lock.unlock();
        }
    }
}