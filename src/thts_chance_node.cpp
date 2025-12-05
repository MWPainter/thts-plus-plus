#include "thts_decision_node.h"

#include "helper_templates.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <cstddef>
#include <functional>
#include <mutex>
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
            ThtsNode(),
            thts_manager(thts_manager),
            state(state),
            action(action),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            parent(parent),
            num_visits(0),
            children(),
            empirical_distribution(),
            local_reward(thts_manager->thts_env()->get_reward_itfc(state,action,*thts_manager->get_thts_context()))
    {
    }

    /**
     * Constructor that skips local_reward initialization.
     * Used by multi-objective subclasses where get_reward_itfc is deleted.
     */
    ThtsCNode::ThtsCNode(
        SkipLocalRewardInit,
        shared_ptr<ThtsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ThtsDNode> parent) :
            ThtsNode(),
            thts_manager(thts_manager),
            state(state),
            action(action),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            parent(parent),
            num_visits(0),
            children(),
            empirical_distribution(),
            local_reward(0.0)
    {
    }

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void ThtsCNode::visit_itfc(ThtsContext& ctx) {
        num_visits += 1;
    }

    /**
     * Updates the empirical distribution map with a given observation.
     * 
     * Args:
     *     observation: The observation to update the empirical distribution with
     */
    void ThtsCNode::update_empirical_distribution(std::shared_ptr<const Observation> observation) {
        empirical_distribution[observation]++;
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
     */
    shared_ptr<ThtsDNode> ThtsCNode::create_child_node_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) 
    {
        if (has_child_node_itfc(observation)) return get_child_node_itfc(observation);

        if (!thts_manager->graph_search) {
            shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
            children[observation] = child_node;
            return child_node;
        }

        DNodeTable& dmap = thts_manager->dmap;
        
        // reading from dnode table
        unique_lock<shared_mutex> writer_lock(thts_manager->dmap_lock);
        // shared_lock<shared_mutex> reader_lock(thts_manager->dmap_lock);
        if (dmap.contains(observation)) {
            shared_ptr<ThtsDNode> child_node = shared_ptr<ThtsDNode>(dmap[observation]);
            children[observation] = child_node;
            return child_node;
        }
        // reader_lock.unlock();

        // writing to dnode table
        shared_ptr<ThtsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
        // unique_lock<shared_mutex> writer_lock(thts_manager->dmap_lock);
        children[observation] = child_node;
        dmap[observation] = child_node;
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

        // Sort children by visit count (so more visit higher/first to see)
        using ObservationNodePair = std::pair<shared_ptr<const Observation>,shared_ptr<ThtsDNode>>;
        vector<ObservationNodePair> children_to_print(children.begin(), children.end());
        std::sort(
            children_to_print.begin(), 
            children_to_print.end(),
            [this](const auto& u, const auto& v) {
                return empirical_distribution.at(u.first) > empirical_distribution.at(v.first);
            }
        );
        
        // print out child trees recursively
        for (ObservationNodePair& key_val_pair : children_to_print) {
            shared_ptr<const Observation> observation = key_val_pair.first;
            ThtsDNode& child_node = *(key_val_pair.second);
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "({" << *observation << "}," << empirical_distribution.at(observation) << ")->";
            child_node.get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
        }

        // Print out closing bracket
        ss << "\n";
        for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "],";
    }
}