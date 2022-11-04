#pragma once

#include "helper_templates.h"
#include "thts_chance_node.h"
#include "thts_decision_node.h"
#include "thts_types.h"

#include <memory>
#include <ostream>
#include <tuple>
#include <unordered_map>


namespace thts {
    // forward declare
    class ThtsDNode;
    class ThtsCNode;

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
     *      dmap:
     *          A transposition table for decision nodes
     *      cmap:
     *          A transposition table for chance nodes
     *      mcts_mode:
     *          If running in mcts_mode (see options above)
     *      use_transposition_table:
     *          If using the transposition tables (i.e. dmap and cmap)
     *      is_two_player_game:
     *          If we are planning for a two player game, rather than a reward maximisation environment
     *      heuristic_fn_ptr:
     *          A pointer to the heuristic function
     *      prior_fn_ptr:
     *          A pointer to the prior (Q-value) function, that returns a map from actions to prior value estimates
     */
    class ThtsManager {

        public:
            DNodeTable dmap;
            CNodeTable cmap;

            bool mcts_mode = true;
            bool use_transposition_table = false;
            bool is_two_player_game = false;

            HeuristicFnPtr heuristic_fn_ptr;
            PriorFnPtr prior_fn_ptr;

            ThtsManager(
                bool mcts_mode=true, 
                bool use_transposition_table=false, 
                bool is_two_player_game=false) :
                    mcts_mode(mcts_mode), 
                    use_transposition_table(use_transposition_table), 
                    is_two_player_game(is_two_player_game) {};
    };
}