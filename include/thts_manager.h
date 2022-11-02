#pragma once

#include "helper_templates.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_types.h"

#include <memory>
#include <ostream>
#include <tuple>
#include <unordered_map>



// /**
//  * Code only compiles if these hash and equal_to definitions are placed here.
//  * TODO: would be more appropraitely places in thts_types.cpp, but generally move somewhere more appropriate
//  */
// namespace std {
//     /**
//      * Implement hash for decision node transposition table keys (see thts_manager.h)
//      */
//     template<>
//     class hash<thts::DNodeIdTuple> {
//         public:
//             size_t operator()(const thts::DNodeIdTuple& dnode_id_tuple) const {
//                 size_t hash_val = 0;
//                 hash_val = thts::helper::hash_combine(hash_val, get<0>(dnode_id_tuple));
//                 return thts::helper::hash_combine(hash_val, get<1>(dnode_id_tuple));
//             }
//     };

//     /**
//      * Implement equal_to for decision node transposition table keys (see thts_manager.h)
//      */
//     template<>
//     class equal_to<thts::DNodeIdTuple> {
//         public:
//             bool operator()(const thts::DNodeIdTuple& lhs, const thts::DNodeIdTuple& rhs) const {
//                 return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs);
//             }
//     };

//     /**
//      * Implement hash for chance node transposition table keys (see thts_manager.h).
//      */
//     template<>
//     class hash<thts::CNodeIdTuple> {
//         public:
//             size_t operator()(const thts::CNodeIdTuple& cnode_id_tuple) const {
//                 size_t hash_val = 0;
//                 hash_val = thts::helper::hash_combine(hash_val, get<0>(cnode_id_tuple));
//                 hash_val = thts::helper::hash_combine(hash_val, get<1>(cnode_id_tuple));
//                 return thts::helper::hash_combine(hash_val, get<2>(cnode_id_tuple));
//             }
//     };

//     /**
//      * Implement equal_to for chance node transposition table keys (see thts_manager.h).
//      */
//     template<>
//     class equal_to<CNodeIdTuple> {
//         public:
//             bool operator()(const CNodeIdTuple& lhs, const CNodeIdTuple& rhs) const {
//                 return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs) && get<2>(lhs) == get<2>(rhs);
//             }
//     };
// }

namespace thts {
    // forward declare
    class ThtsDNode;
    class ThtsCNode;

    // /**
    //  * Typedef for dnode id tuples, for readability.
    //  * First used transposition table implementation, in thts_manager.h, thts_decision_node.h and thts_chance_node.h
    //  */
    // typedef std::tuple<int,std::shared_ptr<const Observation>> DNodeIdTuple;
    // typedef std::unordered_map<DNodeIdTuple,std::shared_ptr<ThtsDNode>> DNodeTable;

    // /**
    //  * Typedef for cnode id tuples, for readability.
    //  * First used transposition table implementation, in thts_manager.h, thts_decision_node.h and thts_chance_node.h
    //  */
    // typedef std::tuple<int,std::shared_ptr<const State>,std::shared_ptr<const Action>> CNodeIdTuple;
    // typedef std::unordered_map<CNodeIdTuple,std::shared_ptr<ThtsCNode>> CNodeTable;

    /**
     * ThtsManager is an object used to manage all the things that need to be 'global' space within Thts.
     * 
     * Primarily a thts manager stores all of the options that thts can be run with (see options section below).
     * 
     * One of the main jobs of the manager is to manage the transposition table(s),
     * 
     * Options:
     *      mcts_mode:
     *          If mcts_mode is true, then only one node is added per trial (and initialised using the heuristic function). 
     *          If mcts_mode is false, then trials are run to completion (until max depth or a sink state is reached).
     *      transposition_table:
     *          Specifies if a transposition table is to be used. Nodes are stored in a table upon creation, keyed by 
     *          (depth, State, optional<Action>) tuples. When creating a new node, we first look if it exists in the table 
     *          already, and if it does we return that instead. This requires State and Action objects to have std::hash and 
     *          std::equal_to definitions.
     *      is_two_player_game:
     *          Specifies if we are planning for a two player game
     * 
     * Member variables:
     *      TODO: write this one 
     *      TODO: check other object docstrings
     *      TODO: document test + fix tests
     *      TODO: fix transposition table
     *      TODO: push code
     *      TODO: gmock + CI
     */
    class ThtsManager {

        public:
            DNodeTable dmap;
            CNodeTable cmap;

            bool mcts_mode = true;
            bool use_transposition_table = false;
            bool is_two_player_game = false;

            ThtsManager(
                bool mcts_mode=true, 
                bool use_transposition_table=false, 
                bool is_two_player_game=false) :
                    mcts_mode(mcts_mode), 
                    use_transposition_table(use_transposition_table), 
                    is_two_player_game(is_two_player_game) {};
    };
}

// namespace std {
//     /**
//      * Hash, equality and stream functions for DNodeIdTuple
//      */
//     template <> struct hash<DNodeIdTuple>;
//     template <> struct equal_to<DNodeIdTuple>;
//     ostream& operator<<(ostream& os, const DNodeIdTuple& tpl);

//     /**
//      * Hash, equality and stream functions for CNodeIdTuple
//      */
//     template <> struct hash<CNodeIdTuple>;
//     template <> struct equal_to<CNodeIdTuple>;
//     ostream& operator<<(ostream& os, const CNodeIdTuple& tpl);
    
//     /**
//      * Output streams for transposition tables
//      */
//     ostream& operator<<(ostream& os, const DNodeTable& tbl);
//     ostream& operator<<(ostream& os, const CNodeTable& tbl);
// }