#include "thts_decision_node.h"

#include "helper_templates.h"
#include "thts_manager.h"

#include <tuple>
#include <utility>

#include <iostream>

using namespace std;
using namespace thts;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    ThtsCNode::ThtsCNode(
        shared_ptr<ThtsManager> thts_manager,
        shared_ptr<ThtsEnv> thts_env,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<ThtsDNode> parent,
        HeuristicFnPtr heuristic_fn_ptr) :
            thts_manager(thts_manager),
            thts_env(thts_env),
            state(state),
            action(action),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            num_visits(0),
            parent(parent),
            heuristic_fn_ptr(heuristic_fn_ptr)  {}

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void ThtsCNode::visit_itfc(ThtsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * Wrapper around 'create_child_node_helper' that include logic for using a transposition table.
     * 
     * If not using a transposition table, we call the helper and put the child in our children map. 
     * If using a transposition table, we first check the transposition table to try get it from there. If it's not in 
     * the table, we make the child and insert it in children and the transposition table.
     */
    shared_ptr<ThtsDNode> ThtsCNode::create_child_node_itfc(shared_ptr<const Observation> observation) {
        if (!thts_manager->use_transposition_table) {
            shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation);
            children[observation] = child_node;
            return child_node;
        }

        DNodeTable& dmap = thts_manager->dmap;
        DNodeIdTuple dnode_id = make_tuple(decision_timestep, observation);
        auto iter = dmap.find(dnode_id);
        if (iter != dmap.end()) {
            shared_ptr<ThtsDNode> child_node = dmap[dnode_id];
            children[observation] = child_node;
            return child_node;
        }

        shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation);
        children[observation] = child_node;
        dmap[dnode_id] = child_node;
        return child_node;
    }

    /**
     * Just passes information out of the thts manager
     */
    bool ThtsCNode::is_two_player_game() {
        return thts_manager->is_two_player_game;
    }

    /**
     * In 2 player games, opponent is the agent going second. If the decision timestep is odd, then this node is an 
     * opponent node. (And we can check for oddness by checking last bit of decision timestep).
     */
    bool ThtsCNode::is_opponent() {
        if (!is_two_player_game()) return false;
        return (decision_timestep & 1) == 1;
    }

    /**
     * Number of children = length of children map
     */
    int ThtsCNode::get_num_children() {
        return children.size();
    }

    /**
     * Has child if it's in the children map. Find returns an iterator pointing at the element found, or the 'end' 
     * iterator if it is not in the map. So if the returned iterator == children.end() then a child doesn't exist for 
     * that action in the children map.
     */
    bool ThtsCNode::has_child_itfc(shared_ptr<const Observation> observation) {
        auto iterator = children.find(observation);
        return iterator != children.end();
    }
    
    /**
     * Just looks up observation in 'children' map.
     */
    shared_ptr<ThtsDNode> ThtsCNode::get_child_node_itfc(shared_ptr<const Observation> observation) {
        return children[observation];
    }

    /**
     * Returns a pretty printing of the node as a string. This is just a wrapper around the helper function. 
     * 
     * The helper function uses a depth with respect to the tree, rather than decision depth, hence why it is multiplied
     * by two (and plus one). A decision depth of zero would print out the child decision nodes still.
     */
    string ThtsCNode::get_pretty_print_string(int depth) {   
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
     * TODO: nice way of only displaying the 5 most visited actions
     */
    void ThtsCNode::get_pretty_print_string_helper(stringstream& ss, int depth, int num_tabs) {
        // Print out this nodes info
        // for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "C(vl=" << get_pretty_print_val() << ",#v=" << num_visits << ")[";

        // print out child trees recursively
        for (pair<const shared_ptr<const Observation>,shared_ptr<ThtsDNode>>& key_val_pair : children) {
            shared_ptr<const Observation> observation = key_val_pair.first;
            shared_ptr<ThtsDNode> child_node = key_val_pair.second;
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "{" << observation << "}->";
            child_node->get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
        }

        // Print out closing bracket
        ss << "\n";
        for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "],";
    }
}