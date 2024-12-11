#include "go/go_state_action.h"

#include "helper_templates.h"
#include "go/go_env.h"

#include "KataGo/cpp/game/board.h"

#include <sstream>

using namespace std;

namespace thts {
    /**
     * Constructor for GoAction
     */
    GoAction::GoAction(short loc) : loc(loc) {}


    /**
     * Gets the x coord for a move
    */
    int GoAction::get_x_coord(int board_size) const {
        if (loc == Board::PASS_LOC) return -1;
        return (loc % (board_size+1)) - 1;
    }

    /**
     * Gets the y coord for a move
    */
    int GoAction::get_y_coord(int board_size) const {
        if (loc == Board::PASS_LOC) return -1;
        return (loc / (board_size+1)) - 1;
    }

    /**
     * GoAction hash
     */
    size_t GoAction::hash() const {
        return std::hash<short>()(loc);
    }

    /**
     * GoAction equals
     */
    bool GoAction::equals(const GoAction& other) const {
        return loc == other.loc;
    }

    /**
     * GoAction equals interface
     */
    bool GoAction::equals_itfc(const Action& other) const {
        try {
            const GoAction& oth = dynamic_cast<const GoAction&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * GoAction get string
     */
    string GoAction::get_pretty_print_string() const {
        stringstream ss;
        // ss << loc;
        ss << "(" << get_x_coord(GoEnv::cur_board_size) << "," << get_y_coord(GoEnv::cur_board_size) << ")";
        return ss.str();
    }




    /**
     * Constructor for GoState
     */
    GoState::GoState(shared_ptr<BoardHistory> board_history) : 
        board_history(board_history), hash_val(compute_hash()), nn_output_cache(make_shared<GoStateNNCache>()) 
    {
        nn_output_cache->has_cached_nn_output = false;
    }

    /**
     * Get the current board for the current state of play
     * 
     * The 'getRecentBoard(x)' function returns the board from 'x' moves ago
     */
    const Board& GoState::get_current_board() const {
        return board_history->getRecentBoard(0);
    }

    /** Get the board history
     * 
    */
    shared_ptr<BoardHistory> GoState::get_board_history() const {
        return board_history;
    }

    /**
     * Get the result of the game +1 for black win, -1 for white win, 0 for draw?
     * Assumes that we know game as ended and been scored when calling
    */
    double GoState::get_result() const {
        double score = get_score();
        if (score > 0.0) return 1.0;
        if (score < 0.0) return -1.0;
        return 0.0;
    }

    /**
     * Gets the score of the game, assumes that we know game as ended and been scored when calling
    */
    double GoState::get_score() const {
        return -1.0 * board_history->finalWhiteMinusBlackScore;
    }

    /**
     * Checks if two boards are equal. Simplified version of KataGo's 'Board::isEqualForTesting'.
     */
    bool GoState::equal_boards(const Board& lhs, const Board& rhs) const {
        if (lhs.x_size != rhs.x_size) return false;
        if (lhs.y_size != rhs.y_size) return false;
        if (lhs.ko_loc != rhs.ko_loc) return false;
        if (lhs.pos_hash != rhs.pos_hash) return false;

        for (int i=0; i<Board::MAX_ARR_SIZE; i++) {
            if (lhs.colors[i] != rhs.colors[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * GoState equals
     */
    bool GoState::equals(const GoState& other) const {
        for (int i=0; i<BoardHistory::NUM_RECENT_BOARDS; i++) {
            if (board_history->presumedNextMovePla != other.board_history->presumedNextMovePla){
                return false;
            }
            const Board& lhs = board_history->getRecentBoard(i);
            const Board& rhs = other.board_history->getRecentBoard(i);
            if (!equal_boards(lhs, rhs)) {
                return false;
            }
        }
        return true;
    }

    /**
     * GoState equals interface
     */
    bool GoState::equals_itfc(const Observation& other) const {
        try {
            const GoState& oth = dynamic_cast<const GoState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Computes hash for a board using combine hash and all of the members checked in 'equal_boards'.
     */
    size_t GoState::hash_board(const Board& board) const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash, board.x_size);
        cur_hash = helper::hash_combine(cur_hash, board.y_size);
        cur_hash = helper::hash_combine(cur_hash, board.ko_loc);
        cur_hash = helper::hash_combine(cur_hash, board.pos_hash.hash0);
        cur_hash = helper::hash_combine(cur_hash, board.pos_hash.hash1);

        for (int i=0; i<Board::MAX_ARR_SIZE; i++) {
            cur_hash = helper::hash_combine(cur_hash, board.colors[i]);
        }
        return cur_hash;
    }

    /**
     * GoState computes hash using the underlying board history. Called at initialisation to set 'hash_val'
     */
    size_t GoState::compute_hash() {
        size_t cur_hash = 0;
        for (int i=0; i<BoardHistory::NUM_RECENT_BOARDS; i++) {
            const Board& board = board_history->getRecentBoard(i);
            size_t board_hash = hash_board(board);
            cur_hash = helper::hash_combine(cur_hash, board_hash);
        }
        cur_hash = helper::hash_combine(cur_hash, board_history->presumedNextMovePla);
        return cur_hash;
    }

    /**
     * Hash function returns cached hash value
     */
    size_t GoState::hash() const {
        return hash_val;
    }

    /**
     * GoState get string
     */
    string GoState::get_pretty_print_string() const {
        return "";
        // stringstream ss;
        // const Board& board = get_current_board();
        // ss << "Go(" << board.x_size << "," << board.y_size << ")[" << board.colors[0];
        // for (int i=1; i<Board::MAX_ARR_SIZE; i++) {
        //     ss << "," << (int)board.colors[i];
        // }
        // ss << "]";
        // return ss.str();
    }    
}

/**
 * Implementations of std::hash and std::equal_to
 */ 
namespace std {
    size_t hash<GoAction>::operator()(const GoAction& action) const {
        return action.hash();
    }

    size_t hash<shared_ptr<const GoAction>>::operator()(const shared_ptr<const GoAction>& action) const {
        return action->hash();
    }
    bool equal_to<GoAction>::operator()(const GoAction& lhs, const GoAction& rhs) const {
        return lhs.equals(rhs);
    }

    bool equal_to<shared_ptr<const GoAction>>::operator()(
        const shared_ptr<const GoAction>& lhs, const shared_ptr<const GoAction>& rhs) const 
    {
        return lhs->equals(*rhs);
    }

    size_t hash<GoState>::operator()(const GoState& state) const {
        return state.hash();
    }

    size_t hash<shared_ptr<const GoState>>::operator()(const shared_ptr<const GoState>& state) const {
        return state->hash();
    }
    bool equal_to<GoState>::operator()(const GoState& lhs, const GoState& rhs) const {
        return lhs.equals(rhs);
    }

    bool equal_to<shared_ptr<const GoState>>::operator()(
        const shared_ptr<const GoState>& lhs, const shared_ptr<const GoState>& rhs) const 
    {
        return lhs->equals(*rhs);
    }
}