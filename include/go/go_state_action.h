#pragma once

#include "thts_types.h"

#include <cstddef>
#include <memory>
#include <string>

#include "KataGo/cpp/game/boardhistory.h"
#include "KataGo/cpp/neuralnet/nneval.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

namespace thts {
    /**
     * Actions in KataGo are represented using the 'Loc' datatype, which is just a typedef for short.
     * 
     * This is basically a copy of IntAction from thts_types.h.
     * 
     * Member variables:
     *      loc: The KataGo Loc value, corresponding to a move on a go board 
     */
    class GoAction : public Action {
        public:
            short loc;

            GoAction(short loc);

            int get_x_coord(int board_size) const;
            int get_y_coord(int board_size) const;

            virtual std::size_t hash() const;
            bool equals(const GoAction& other) const;
            virtual bool equals_itfc(const Action& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * Indirection for cache values, so can update cache values in a const GoState object
    */
    struct GoStateNNCache {
        bool has_cached_nn_output;
        double cached_white_win_prob;
        double cached_white_lose_prob;
        double cached_no_result_prob;
        double cached_value;
        double cached_white_score_mean;
        float cached_policy[NNPos::MAX_NN_POLICY_SIZE];
        int nn_x_len;
        int nn_y_len;
    };

    /**
     * GoState is a wrapper around KataGo's board history object.
     * 
     * Member variables:
     *      board_history: The underlying KataGo board history for this state
     *      hash_val: A cached value for the hash value
     *      nn_output_cache: A cache for the nn outputs for this state (to avoid having to reevaluate the nn)
     */
    class GoState : public State {
        public:
            std::shared_ptr<BoardHistory> board_history;
            const size_t hash_val;
            std::shared_ptr<GoStateNNCache> nn_output_cache;

            GoState(std::shared_ptr<BoardHistory> board_history);

            const Board& get_current_board() const;
            std::shared_ptr<BoardHistory> get_board_history() const;

            /**
             * Get the result of the game +1 for black win, -1 for white win, 0 for draw?
             * Assumes that we know game as ended and been scored when calling
            */
            double get_result() const;

            /**
             * Gets the score of the game, assumes that we know game as ended and been scored when calling
            */
            double get_score() const;

            /**
             * Interface required by 'State' superclass.
             * (And helper functions to compute them).)
             */
            bool equal_boards(const Board& lhs, const Board& rhs) const;
            bool equals(const GoState& other) const;
            virtual bool equals_itfc(const Observation& other) const;

            std::size_t hash_board(const Board& board) const;
            std::size_t compute_hash();
            virtual std::size_t hash() const;

            virtual std::string get_pretty_print_string() const;
    };
}

/**
 * std::hash and std::equal_to declarations for GoAction and GoState
 */
namespace std {
    using namespace thts;

    template <> 
    struct hash<GoState> {
        size_t operator()(const GoState&) const;
    };

    template <> 
    struct hash<shared_ptr<const GoState>> {
        size_t operator()(const shared_ptr<const GoState>&) const;
    };

    template <> 
    struct equal_to<GoState> {
        bool operator()(const GoState&, const GoState&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const GoState>> {
        bool operator()(const shared_ptr<const GoState>&, const shared_ptr<const GoState>&) const;
    };

    template <> 
    struct hash<GoAction> {
        size_t operator()(const GoAction&) const;
    };

    template <> 
    struct hash<shared_ptr<const GoAction>> {
        size_t operator()(const shared_ptr<const GoAction>&) const;
    };

    template <> 
    struct equal_to<GoAction> {
        bool operator()(const GoAction&, const GoAction&) const;
    };

    template <> 
    struct equal_to<shared_ptr<const GoAction>> {
        bool operator()(const shared_ptr<const GoAction>&, const shared_ptr<const GoAction>&) const;
    };
}