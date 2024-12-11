#include "thts_decision_node.h"

#include "helper_templates.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

using namespace std;
using namespace thts;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    ThtsCNode::ThtsCNode(
        shared_ptr<ThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ThtsDNode> parent) :
            node_lock(),
            thts_manager(thts_manager),
            state(state),
            action(action),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            parent(parent),
            num_visits(0),
            next_state_distr(thts_manager->thts_env->get_transition_distribution_itfc(state,action)),
            child_constructed()
    {
        // TODO: when integrate in main, use the thread id (put it in context), and use that id for the 
        // 'unconstructed' value. This way the thread can take ownership of creating the chance node and decision 
        // node in the deterministic setting. For now, the race condition is unlikely and worst case is that an 
        // occasional search thread has to wait
        for (pair<shared_ptr<const State>,double> pr : *next_state_distr) {
            child_constructed.emplace(pr.first,DNODE_STATE_UNCONSTRUCTED);
        }
    }

    /**
     * Aquires the lock for this node.
     */
    void ThtsCNode::lock() { 
        node_lock.lock(); 
    }

    /**
     * Releases the lock for this node.
     */
    void ThtsCNode::unlock() { 
        node_lock.unlock(); 
    }

    /**
     * Gets a reference to the lock for this node (so can use in a lock_guard for example)
     */
    std::mutex& ThtsCNode::get_lock() { 
        return node_lock; 
    }

    /**
     * Helper function to lock all children nodes.
     */
    void ThtsCNode::lock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->lock();
        }
    }

    /**
     * Helper function to unlock all children nodes.
     */
    void ThtsCNode::unlock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->unlock();
        }
    }

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void ThtsCNode::visit_itfc(ThtsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * Wrapper around 'create_child_node_helper' that include logic for using a transposition table.
     * 
     * If child already exists then just return it.
     * 
     * If not using a transposition table, we call the helper and put the child in our children map. 
     * If using a transposition table, we first check the transposition table to try get it from there. If it's not in 
     * the table, we make the child and insert it in children and the transposition table.
     * 
     * Additionally, we protect accessing 'dmap[dnode_id]' with the mutex 'thts_manager->dmap_mutexes[mutex_indx]' 
     * where 'mutex_indx = hash(dnode_id) % thts_manager->dmap_mutexes.size()', by locking it using a lock_guard.
     * 
     * Updates for the avoid_selecting_children_under_construction mode, updates the construction state when claiming 
     * the construction and after putting it in the child map. If we fail to update the construction state, it means 
     * that another thread either is constructing or has constructed the child. We can safely return a nullptr, and 
     * the select action methods should loop if the action/next state corresponds to a child that is under construciton.
     */
    shared_ptr<ThtsDNode> ThtsCNode::create_child_node_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) 
    {
        if (has_child_node_itfc(observation)) return get_child_node_itfc(observation);

        if (!thts_manager->use_transposition_table) {
            if (thts_manager->avoid_selecting_children_under_construction) {
                shared_ptr<const State> state = static_pointer_cast<const State>(observation);
                                                                    // N.B. this makes it no longer work for Observation != State,
                bool success = set_child_under_construction(state); // but that's never the case in this branch 
                                                                    // N.B.B. When integrate to main, fix this
                if (!success) return nullptr;
            }
            shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
            children[observation] = child_node;
            if (thts_manager->avoid_selecting_children_under_construction) {
                shared_ptr<const State> state = static_pointer_cast<const State>(observation);
                                                                    // N.B. this makes it no longer work for Observation != State,
                bool success = set_child_constructed(state); // but that's never the case in this branch 
                                                                    // N.B.B. When integrate to main, fix this
                if (!success) throw runtime_error("Failed to finish construction after claiming construction.");
            }
            return child_node;
        }

        DNodeTable& dmap = thts_manager->dmap;
        DNodeIdTuple dnode_id = make_tuple(decision_timestep, observation);

        int mutex_indx = 0;
        if (thts_manager->dmap_mutexes.size() > 1) {
            size_t tpl_hash = hash<DNodeIdTuple>()(dnode_id);
            mutex_indx = tpl_hash % thts_manager->dmap_mutexes.size();
        }

        lock_guard<mutex> lg(thts_manager->dmap_mutexes[mutex_indx]);

        auto iter = dmap.find(dnode_id);
        if (iter != dmap.end()) {
            try {
                shared_ptr<ThtsDNode> child_node = shared_ptr<ThtsDNode>(dmap[dnode_id]);
                children[observation] = child_node;
                return child_node;
            } catch (const bad_weak_ptr& e) {
                // from my understanding, we should never call this when the pointers don't exist?
                // think there is something happening that don't fully understand right now
                // think that if get a bad_weak_ptr, then should be fine to continue onto making the node and re-insert
            }
        }

        if (thts_manager->avoid_selecting_children_under_construction) {
            shared_ptr<const State> state = static_pointer_cast<const State>(observation);
                                                                // N.B. this makes it no longer work for Observation != State,
            bool success = set_child_under_construction(state); // but that's never the case in this branch 
                                                                // N.B.B. When integrate to main, fix this
            if (!success) return nullptr;
        }
        shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
        children[observation] = child_node;
        dmap[dnode_id] = child_node;
        if (thts_manager->avoid_selecting_children_under_construction) {
            shared_ptr<const State> state = static_pointer_cast<const State>(observation);
            bool success = set_child_constructed(state);    /// N.B.B.B. SAME prob here
            if (!success) throw runtime_error("Failed to finish construction after claiming construction.");
        }
        return child_node;
    }

    /**
     * Just passes information out of the thts manager
     */
    bool ThtsCNode::is_two_player_game() const {
        return thts_manager->is_two_player_game;
    }

    /**
     * In 2 player games, opponent is the agent going second. If the decision timestep is odd, then this node is an 
     * opponent node. (And we can check for oddness by checking last bit of decision timestep).
     */
    bool ThtsCNode::is_opponent() const {
        // TODO: when integrate in main
        if (!is_two_player_game()) return false;
        return (decision_timestep & 1) == 1;
    }

    /**
     * Number of children = length of children map
     */
    int ThtsCNode::get_num_children() const {
        return children.size();
    }

    /**
     * Has child if it's in the children map. Find returns an iterator pointing at the element found, or the 'end' 
     * iterator if it is not in the map. So if the returned iterator == children.end() then a child doesn't exist for 
     * that action in the children map.
     */
    bool ThtsCNode::has_child_node_itfc(shared_ptr<const Observation> observation) const {
        auto iterator = children.find(observation);
        return iterator != children.end();
    }
    
    /**
     * Just looks up observation in 'children' map.
     */
    shared_ptr<ThtsDNode> ThtsCNode::get_child_node_itfc(shared_ptr<const Observation> observation) const {
        return children.at(observation);
    }

    /**
     * Returns a pretty printing of the node as a string. This is just a wrapper around the helper function. 
     * 
     * The helper function uses a depth with respect to the tree, rather than decision depth, hence why it is multiplied
     * by two (and plus one). A decision depth of zero would print out the child decision nodes still.
     */
    string ThtsCNode::get_pretty_print_string(int depth) const {   
        int num_tabs = 0;
        stringstream ss;
        get_pretty_print_string_helper(ss, 2*depth+1, num_tabs);
        return ss.str();
    }

    /**
     * Recursively pretty prints a tree and the values given by nodes 'get_pretty_print_val' functions.
     * 
     * Should be a one-and-done function that can be reused. It's pretty much all just building a string that lays out 
     * nodes 'get_pretty_print_val' values in a nice format. 
     * 
     * TODO: add nice way of only displaying the X most sampled outcomes
     */
    void ThtsCNode::get_pretty_print_string_helper(stringstream& ss, int depth, int num_tabs) const {
        // Print out this nodes info
        ss << "C(vl=" << get_pretty_print_val() << ",#v=" << num_visits << ")[";

        // print out child trees recursively
        for (const pair<const shared_ptr<const Observation>,shared_ptr<ThtsDNode>>& key_val_pair : children) {
            const Observation& observation = *(key_val_pair.first);
            ThtsDNode& child_node = *(key_val_pair.second);
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "{" << observation << "}->";
            child_node.get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
        }

        // Print out closing bracket
        ss << "\n";
        for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "],";
    }
    
    /**
     * Tries to set that the child is going to be constructed (and is 'under construction'). Returns false if it fails 
     * to update the state
    */
    bool ThtsCNode::set_child_under_construction(shared_ptr<const State> state) {
        // TODO: when integrate in main, use the thread id (put it in context), and use that id for the 
        // 'unconstructed' value. This way the thread can take ownership of creating the chance node and decision 
        // node in the deterministic setting. For now, the race condition is unlikely and worst case is that an 
        // occasional search thread has to wait
        int unconstructed = DNODE_STATE_UNCONSTRUCTED;
        int under_construciton = DNODE_STATE_UNDER_CONSTRUCTION;
        return child_constructed.at(state).compare_exchange_strong(unconstructed, under_construciton);
    }

    /**
     * Tries to set that child is constructed. Returns false if it fails to update.
    */
    bool ThtsCNode::set_child_constructed(shared_ptr<const State> state) {
        int under_construciton = DNODE_STATE_UNDER_CONSTRUCTION;
        int constructed = DNODE_STATE_CONSTRUCTED;
        return child_constructed.at(state).compare_exchange_strong(under_construciton, constructed);
    }
    
    /**
     * When running in avoid_selecting_children_under_construction mode, this is just a null ptr check
     * Otherwise it reutns true if its a nullptr or the child decision node is under construction
    */
    bool ThtsCNode::is_nullptr_or_should_skip_under_construction_child(shared_ptr<const State> state) {
        // node_lock.unlock();
        // std::this_thread::yield();
        // node_lock.lock();

        if (state == nullptr) return true;
        if (!thts_manager->avoid_selecting_children_under_construction) return false;
        int node_state = child_constructed.at(state).load();
        if (node_state == DNODE_STATE_UNDER_CONSTRUCTION) return true;
        return false;
    }
}