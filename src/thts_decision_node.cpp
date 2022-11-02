#include "thts_decision_node.h"

#include "helper_templates.h"
#include "thts_manager.h"

#include <tuple>
#include <utility>

using namespace std;
using namespace thts;

// // TODO: move to thts_manager.cpp?
// namespace std {
//     /**
//      * Implement hash for chance node transposition table keys (see thts_manager.h).
//      */
//     class hash<CNodeIdTuple> {
//         public:
//             size_t operator()(const CNodeIdTuple& cnode_id_tuple) const {
//                 size_t hash_val = 0;
//                 hash_val = helper::hash_combine(hash_val, get<0>(cnode_id_tuple));
//                 return helper::hash_combine(hash_val, get<1>(cnode_id_tuple));
//                 return helper::hash_combine(hash_val, get<2>(cnode_id_tuple));
//             }
//     };

//     /**
//      * Implement equal_to for chance node transposition table keys (see thts_manager.h).
//      */
//     class equal_to<CNodeIdTuple> {
//         public:
//             bool operator()(const CNodeIdTuple& lhs, const CNodeIdTuple& rhs) const {
//                 return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs) && get<2>(lhs) == get<2>(rhs);
//             }
//     };
// }


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    ThtsDNode::ThtsDNode(
        shared_ptr<ThtsManager> thts_manager,
        shared_ptr<ThtsEnv> thts_env,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<ThtsCNode> parent,
        HeuristicFnPtr heuristic_fn_ptr,
        PriorFnPtr prior_fn_ptr) :
            thts_manager(thts_manager),
            thts_env(thts_env),
            state(state),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            num_visits(0),
            parent(parent),
            heuristic_fn_ptr(heuristic_fn_ptr),
            prior_fn_ptr(prior_fn_ptr) {}

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void ThtsDNode::visit_itfc(ThtsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * Wrapper around 'create_child_node_helper' that include logic for using a transposition table.
     * 
     * If not using a transposition table, we call the helper and put the child in our children map. 
     * If using a transposition table, we first check the transposition table to try get it from there. If it's not in 
     * the table, we make the child and insert it in children and the transposition table.
     */
    shared_ptr<ThtsCNode> ThtsDNode::create_child_node_itfc(shared_ptr<const Action> action) {
        if (!thts_manager->use_transposition_table) {
            shared_ptr<ThtsCNode> child_node = create_child_node_helper_itfc(action);
            children[action] = child_node;
            return child_node;
        }

        CNodeTable& cmap = thts_manager->cmap;
        CNodeIdTuple cnode_id = make_tuple(decision_timestep, state, action);
        auto iter = cmap.find(cnode_id);
        if (iter != cmap.end()) {
            shared_ptr<ThtsCNode> child_node = cmap[cnode_id];
            children[action] = child_node;
            return child_node;
        }

        shared_ptr<ThtsCNode> child_node = create_child_node_helper_itfc(action);
        children[action] = child_node;
        cmap[cnode_id] = child_node;
        return child_node;
    }

    /**
     * A decision node is the root node iff the decision depth is 0.
     * 
     * TODO: this current implementation will lead to bugs if trying to reuse trees. Consider changing implementation 
     * to use parent. And make sure root node doesn't have a parent in thts routine.
     */
    bool ThtsDNode::is_root_node() {
        return decision_depth == 0;
    }

    /**
     * Just passes information out of the thts manager 
     */
    bool ThtsDNode::is_two_player_game() {
        return thts_manager->is_two_player_game;
    }

    /**
     * In 2 player games, opponent is the agent going second. If the decision timestep is odd, then this node is an 
     * opponent node. (And we can check for oddness by checking last bit of decision timestep).
     */
    bool ThtsDNode::is_opponent() {
        if (!is_two_player_game()) return false;
        return (decision_timestep & 1) == 1;
    }

    /**
     * Number of children = length of children map.
     */
    int ThtsDNode::get_num_children() {
        return children.size();
    }

    /**
     * Has child if it's in the children map. Find returns an iterator pointing at the element found, or the 'end' 
     * iterator if it is not in the map. So if the returned iterator == children.end() then a child doesn't exist for 
     * that action in the children map.
     */
    bool ThtsDNode::has_child_itfc(shared_ptr<const Action> action) {
        auto iterator = children.find(action);
        return iterator != children.end();
    }
    
    /**
     * Just looks up action in 'children' map.
     */
    shared_ptr<ThtsCNode> ThtsDNode::get_child_node_itfc(shared_ptr<const Action> action) {
        return children[action];
    }

    /**
     * Returns a pretty printing of the node as a string. This is just a wrapper around the helper function. 
     * 
     * The helper function uses a depth with respect to the tree, rather than decision depth, hence why it is multiplied
     * by two.
     */
    string ThtsDNode::get_pretty_print_string(int depth) {   
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
     * TODO: add nice way of only displaying the 5 most visited actions
     */
    void ThtsDNode::get_pretty_print_string_helper(stringstream& ss, int depth, int num_tabs) {
        // Print out this nodes info
        // for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "D(vl=" << get_pretty_print_val() << ",#v=" << num_visits << ")[";

        // Base case
        if (depth == 0) {
            if (!is_leaf()) ss << "...";
            ss << "],";
            return;
        }

        // print out child trees recursively
        for (pair<const shared_ptr<const Action>,shared_ptr<ThtsCNode>>& key_val_pair : children) {
            shared_ptr<const Action> action = key_val_pair.first;
            shared_ptr<ThtsCNode> child_node = key_val_pair.second;
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "\"" << action << "\"->";
            child_node->get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
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
        throw 0;
    }

    /**
     * Saves a tree to a given filename
     * TODO: implement, and remove throwing exception
     */
    bool ThtsDNode::save(string& filename) {
        throw 0;
    }
}