#include "thts_decision_node.h"

#include "helper_templates.h"
#include "thts_manager.h"

#include <stdexcept>
#include <tuple>
#include <utility>

using namespace std;
using namespace thts;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     * 
     * Nuance use of heuristic value is to enforce nodes for sink states to have a value of zero
     */
    ThtsDNode::ThtsDNode(
        shared_ptr<ThtsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ThtsCNode> parent) :
            node_lock(),
            thts_manager(thts_manager),
            state(state),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            parent(parent),
            num_visits(0),
            heuristic_value(0.0)
    {
        if (thts_manager->heuristic_fn != nullptr && !thts_manager->thts_env->is_sink_state_itfc(state)) {
            heuristic_value = thts_manager->heuristic_fn(state, thts_manager->thts_env);
        }
    }

    /**
     * Aquires the lock for this node.
     */
    void ThtsDNode::lock() 
    { 
        node_lock.lock(); 
    }

    /**
     * Releases the lock for this node.
     */
    void ThtsDNode::unlock() 
    { 
        node_lock.unlock(); 
    }

    /**
     * Gets a reference to the lock for this node (so can use in a lock_guard for example)
     */
    std::mutex& ThtsDNode::get_lock() 
    { 
        return node_lock; 
    }

    /**
     * Helper function to lock all children nodes.
     */
    void ThtsDNode::lock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->lock();
        }
    }

    /**
     * Helper function to unlock all children nodes.
     */
    void ThtsDNode::unlock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->unlock();
        }
    }

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void ThtsDNode::visit_itfc(ThtsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * This node is a sink node iff the state corresponds to a sink state
     */
    bool ThtsDNode::is_sink() const {
        return thts_manager->thts_env->is_sink_state_itfc(state);
    }

    /**
     * This node is a leaf node iff (it is a sink node or it is at the maximum decision depth)
     */
    bool ThtsDNode::is_leaf() const {
        return is_sink() || decision_depth >= thts_manager->max_depth;
    }

    /**
     * Wrapper around 'create_child_node_helper' that include logic for using a transposition table.
     * 
     * If child already exists then just return it.
     * 
     * If not using a transposition table, we call the helper and put the child in our children map. 
     * 
     * As transposition table is implemented for decision nodes, we don't need to use one for chance nodes. If two 
     * chance nodes would be transpositions, then their parent (decision) nodes would be transpositions!
     */
    shared_ptr<ThtsCNode> ThtsDNode::create_child_node_itfc(shared_ptr<const Action> action) {
        if (has_child_node_itfc(action)) return get_child_node_itfc(action);
        shared_ptr<ThtsCNode> child_node = create_child_node_helper_itfc(action);
        children[action] = child_node;
        return child_node;
    }

    /**
     * A decision node is the root node iff the decision depth is 0.
     * 
     * TODO: this current implementation will lead to bugs if trying to reuse trees. Consider changing implementation 
     * to use parent. And make sure root node doesn't have a parent in thts routine.
     */
    bool ThtsDNode::is_root_node() const {
        return decision_depth == 0;
    }

    /**
     * Just passes information out of the thts manager 
     */
    bool ThtsDNode::is_two_player_game() const {
        return thts_manager->is_two_player_game;
    }

    /**
     * In 2 player games, opponent is the agent going second. If the decision timestep is odd, then this node is an 
     * opponent node. (And we can check for oddness by checking last bit of decision timestep).
     */
    bool ThtsDNode::is_opponent() const {
        if (!is_two_player_game()) return false;
        return (decision_timestep & 1) == 1;
    }

    /**
     * Gets the number of times that the node has been visited
    */
    int ThtsDNode::get_num_visits() const {
        return num_visits;
    }

    /**
     * Number of children = length of children map.
     */
    int ThtsDNode::get_num_children() const {
        return children.size();
    }

    /**
     * Has child if it's in the children map. Find returns an iterator pointing at the element found, or the 'end' 
     * iterator if it is not in the map. So if the returned iterator == children.end() then a child doesn't exist for 
     * that action in the children map.
     */
    bool ThtsDNode::has_child_node_itfc(shared_ptr<const Action> action) const {
        auto iterator = children.find(action);
        return iterator != children.end();
    }
    
    /**
     * Just looks up action in 'children' map.
     */
    shared_ptr<ThtsCNode> ThtsDNode::get_child_node_itfc(shared_ptr<const Action> action) const {
        return children.at(action);
    }

    /**
     * Returns a pretty printing of the node as a string. This is just a wrapper around the helper function. 
     * 
     * The helper function uses a depth with respect to the tree, rather than decision depth, hence why it is multiplied
     * by two.
     */
    string ThtsDNode::get_pretty_print_string(int depth) const {   
        int num_tabs = 0;
        stringstream ss;
        get_pretty_print_string_helper(ss, 2*depth, num_tabs);
        return ss.str();
    }

    /**
     * Recursively pretty prints a tree and the values given by nodes 'get_pretty_print_val' functions.
     * 
     * Should be a one-and-done function that can be reused. It's pretty much all just building a string that lays out 
     * nodes 'get_pretty_print_val' values in a nice format. 
     * 
     * TODO: add nice way of only displaying the X most visited actions
     */
    void ThtsDNode::get_pretty_print_string_helper(stringstream& ss, int depth, int num_tabs) const {
        // Print out this nodes info
        ss << "D(vl=" << get_pretty_print_val() << ",#v=" << num_visits << ")[";

        // Base case 
        if (depth == 0) {
            if (!is_leaf()) ss << "...";
            ss << "],";
            return;
        }

        // print out child trees recursively
        for (const pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& key_val_pair : children) {
            const Action& action = *(key_val_pair.first);
            ThtsCNode& child_node = *(key_val_pair.second);
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "\"" << action << "\"->";
            child_node.get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
        }

        // Print out closing bracket
        ss << "\n";
        for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "],";
    }

    /**
     * Loads a tree from filename
     * TODO: implement, and remove throwing exception
     */
    shared_ptr<ThtsDNode> ThtsDNode::load(string& filename) {
        throw runtime_error("Load function not implemented");
    }

    /**
     * Saves a tree to a given filename
     * TODO: implement, and remove throwing exception
     */
    bool ThtsDNode::save(string& filename) const {
        throw runtime_error("Save function not implemented");
    }
}