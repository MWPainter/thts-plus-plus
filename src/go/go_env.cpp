#include "go/go_env.h"

#include "KataGo/cpp/game/board.h"
#include "KataGo/cpp/program/setup.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

using namespace std; 

namespace thts {

    /**
     * Initialise class variable
    */
    int GoEnv::cur_board_size = 19;
    
    /**
     * Constructor
     */
    GoEnv::GoEnv(
        int board_size, 
        float komi, 
        bool use_nn_eval, 
        string nn_eval_rand_seed, 
        float nn_temp, 
        double reward_scale, 
        double dynamic_score_center) :
            ThtsEnv(true),
            board_size(board_size),
            komi(komi),
            init_board(board_size, board_size),
            rules(
                Rules::KO_POSITIONAL,   // ko rule
                Rules::SCORING_AREA,    // scoring rule
                Rules::TAX_NONE,        // tax rule
                false,                  // multi stone suicidelegal
                false,                  // has button
                Rules::WHB_ZERO,        // white handicap (zero)
                false,                  // friendly pass ok
                komi),                  // komi (white handicap in points at end of game)
            init_state(make_shared<GoState>(make_shared<BoardHistory>(
                init_board,             // board
                P_BLACK,                // black starts
                rules,                  // rules we will play with
                0))),                   // encore phase (no idea what this is, only relevant in territory scoring, not area)
            nn_eval(nullptr),
            _cfg(),
            _logger(),
            nn_temp(nn_temp),
            reward_scale(reward_scale),
            dynamic_score_center(dynamic_score_center)
    {
        if (use_nn_eval) {
            string model_filename = NN_FILENAME;
            const string expectedSha256 = "";
            Rand rand(nn_eval_rand_seed);
            int max_concurrent_evals = 256;         
            int expected_concurrent_evals = 128;         
            int default_max_batch_size = 128;       
            bool require_exact_nn_len = true;
            
            nn_eval = Setup::initializeNNEvaluator(
                model_filename,                         // model name
                model_filename,                         // model filename
                expectedSha256,                         // ?, always passed in "" in KataGo so meh
                _cfg,                                   // config parser, don't think we care about this here...
                _logger,                                // logger, don't think we care about this here either...
                rand,                                   // don't know why need random number gen? but okie
                max_concurrent_evals,                   // increase if ever want to run multiple threads in seach, 1 ok for now
                expected_concurrent_evals,              // same as above
                board_size,                             // max X board size used (we only use one)
                board_size,                             // max Y board size used (we only use one)
                default_max_batch_size,                 // presumably default == -1 means any size
                require_exact_nn_len,                   // essentially we expect all nn calls to use max x and max y size
                Setup::SETUP_FOR_MATCH                  // setup for match seems like the best option for running in eval mode
            );
            nn_eval->setDoRandomize(false);

            // set global board size variable
            cur_board_size = board_size;
        }
    }

    /**
     * Destructor
     */
    GoEnv::~GoEnv() {
        if (nn_eval != nullptr) {
            delete nn_eval;
        }
    }

    NNEvaluator* GoEnv::get_nn_eval() {
        return nn_eval;
    }

    Logger* GoEnv::get_logger() {
        return &_logger;
    }

    /**
     * Get (cached) initial state
     */
    shared_ptr<const GoState> GoEnv::get_initial_state() const {
        return init_state;
    }

    /**
     * Board history contains if the game has finished or not
     */
    bool GoEnv::is_sink_state(shared_ptr<const GoState> state) const {
        return state->board_history->isGameFinished;
    }

    /**
     * Iterates through all possible 'loc' values, and checks if the move is legal with 'isLegal'.
     */
    shared_ptr<GoActionVector> GoEnv::get_valid_actions(shared_ptr<const GoState> state) const {
        if (is_sink_state(state)) {
            return make_shared<GoActionVector>();
        }

        shared_ptr<GoState> _state = const_pointer_cast<GoState>(state);
        const Board& board = _state->get_current_board();
        Player player = state->board_history->presumedNextMovePla;
        
        shared_ptr<GoActionVector> actions = make_shared<GoActionVector>();
        for (short loc=0; loc<Board::MAX_ARR_SIZE; loc++) {
            if (state->board_history->isLegal(board, loc, player)) {
                actions->push_back(make_shared<GoAction>(loc));
            }
        }

        return actions;
    }

    /**
     * As deterministic, just 'sample' the move, and then make a delta distribution.
     */
    shared_ptr<GoStateDistr> GoEnv::get_transition_distribution(
        shared_ptr<const GoState> state, shared_ptr<const GoAction> action) const 
    {
        shared_ptr<const GoState> new_state = sample_transition_distribution(state, action);
        shared_ptr<GoStateDistr> transition_distribution = make_shared<GoStateDistr>(); 
        transition_distribution->insert_or_assign(new_state, 1.0);
        return transition_distribution;
    }

    /**
     * Make a move.
     * 
     * We make a new history object (copies from the one in 'state'), and creates the new state object. 
     * 
     * Then we make a copy of the current board, which will be edited in the make board move function, and then copied 
     * again into the new history object.
     * 
     * Finally we add the const with a const cast.
     */
    shared_ptr<const GoState> GoEnv::sample_transition_distribution(
        shared_ptr<const GoState> state, shared_ptr<const GoAction> action) const 
    {
        shared_ptr<BoardHistory> new_board_history = make_shared<BoardHistory>(*(state->board_history));
        shared_ptr<GoState> new_state = make_shared<GoState>(new_board_history);
        Board board(new_state->get_current_board());
        Player player = new_board_history->presumedNextMovePla;
        new_board_history->makeBoardMoveAssumeLegal(
            board, action->loc, player, nullptr);
        return const_pointer_cast<const GoState>(new_state);
    }

    /**
     * Rewards are 1.0 when black wins, and -1.0 when white wins
     */
    double GoEnv::get_reward(
        shared_ptr<const GoState> state, 
        shared_ptr<const GoAction> action, 
        shared_ptr<const GoState> observation) const 
    {
        if (observation == nullptr) {
            observation = sample_transition_distribution(state,action);
        }
        if (!is_sink_state(observation)) return 0.0;
        if (observation->board_history->winner == P_BLACK) return 1.0 * reward_scale;
        if (observation->board_history->winner == P_WHITE) return -1.0 * reward_scale;
        return 0.0;
    }

    /**
     * 
    */
    void GoEnv::update_dynamic_score_center_for_root_state(shared_ptr<const GoState> state) {
        if (!state->nn_output_cache->has_cached_nn_output) {
            fill_cached_values_with_nn_output(state);
        }
        // copy computing a 'dynamic' score from katago (just copied params from 
        // https://github.com/lightvector/KataGo/blob/2fbfb5defc3238af689eb10b9dfdd8766244c16a/cpp/configs/training/selfplay8mainb18.cfg)
        double white_score_mean = state->nn_output_cache->cached_white_score_mean;
        dynamic_score_center = white_score_mean * (1.0 - 0.25);
        Board cur_board(state->get_current_board());
        double cap = sqrt(cur_board.x_size * cur_board.y_size) * 0.5;
        if (dynamic_score_center > white_score_mean + cap) {
            dynamic_score_center = white_score_mean + cap;
        } else if (dynamic_score_center < white_score_mean - cap) {
            dynamic_score_center = white_score_mean - cap;
        }
    }

    /**
     * Interface with KataGo neural net
     */
    void GoEnv::fill_cached_values_with_nn_output(shared_ptr<const GoState> state) {
        // get nn output
        Board cur_board(state->get_current_board());
        const BoardHistory& board_history = *(state->board_history);
        Player pla = board_history.presumedNextMovePla;
        MiscNNInputParams input_params;
        input_params.symmetry = NNInputs::SYMMETRY_ALL;
        NNResultBuf buf;
        bool skip_cache = true;
        bool include_owner_map = false;

        nn_eval->evaluate(cur_board, board_history, pla, input_params, buf, skip_cache, include_owner_map);
        NNOutput& nn_output = *(buf.result);

        // compute value from win prob + score
        double win_value = reward_scale * nn_output.whiteLossProb - reward_scale * nn_output.whiteWinProb;

        // Copy compouting a 'static' score from katago
        double white_score_mean = nn_output.whiteScoreMean;
        double white_score_mean_sq = white_score_mean * white_score_mean;
        double white_score_std_dev = ScoreValue::getScoreStdev(white_score_mean, white_score_mean_sq);
        double white_score_value = ScoreValue::expectedWhiteScoreValue(
            white_score_mean, white_score_std_dev, 0.0, 2.0, cur_board);
        double score_value = -reward_scale * white_score_value;

        // 'dynamic' score from katago
        double dynamic_score_value = ScoreValue::expectedWhiteScoreValue(
            white_score_mean, white_score_std_dev, dynamic_score_center, 0.5, cur_board);


        // double value = 0.25 * win_value + 0.75 * score_value;
        // double value = 0.5 * win_value + 0.5 * score_value;
        // double value = 0.625 * win_value + 0.375 * score_value;
        // double value = 0.75 * win_value + 0.25 * score_value;
        // double value = 0.875 * win_value + 0.125 * score_value;

        // copied params from same link in comment in 'update_dynmaic_score_center_for_state'
        // Assumed that the values are in the range [-1,1], and normalised the weights to 1.0 total
        double value = (1.0 * win_value + 0.1 * score_value + 0.3 * dynamic_score_value) / 1.4;

        // normalise policy
        // find max prob
        // compute rel_prob = (prob/max_prob)^(1/temp), while zeroing out any non-positive probs
        // N.B. that dividing whole distr by same number keeps the same distribution 
        // normalise
        double max_prob = 0.0;
        for (int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
            double prob = nn_output.policyProbs[i];
            if (prob > max_prob) max_prob = prob;
        }

        double sum_probs = 0.0;
        double log_max_prob = log(max_prob);
        for (int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
            double prob = nn_output.policyProbs[i];
            if (prob > 0.0) {
                nn_output.policyProbs[i] = exp((log(prob) - log_max_prob) / nn_temp);
            } else {
                nn_output.policyProbs[i] = 0.0;
            }
            sum_probs += nn_output.policyProbs[i];
        }

        for (int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
            nn_output.policyProbs[i] /= sum_probs;
        }
        
        // store in cache
        state->nn_output_cache->has_cached_nn_output = true;

        state->nn_output_cache->cached_white_win_prob = nn_output.whiteWinProb;
        state->nn_output_cache->cached_white_lose_prob = nn_output.whiteLossProb;
        state->nn_output_cache->cached_no_result_prob = nn_output.whiteNoResultProb;

        state->nn_output_cache->cached_value = value;
        state->nn_output_cache->cached_white_score_mean = white_score_mean;

        state->nn_output_cache->nn_x_len = nn_output.nnXLen;
        state->nn_output_cache->nn_y_len = nn_output.nnYLen;

        for (int i=0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
            state->nn_output_cache->cached_policy[i] = nn_output.policyProbs[i];
        }
    }
        
    /**
     * Calls neural net, unless already have result cached, and returns a heuristic value for black winning
     */
    double GoEnv::get_heuristic_val_from_nn(shared_ptr<const GoState> state) {
        if (!state->nn_output_cache->has_cached_nn_output) {
            fill_cached_values_with_nn_output(state);
        }
        return state->nn_output_cache->cached_value; 
    }

    double GoEnv::get_heuristic_val_from_nn_itfc(shared_ptr<const State> state) {
        shared_ptr<const GoState> go_state = static_pointer_cast<const GoState>(state);
        return get_heuristic_val_from_nn(go_state);
    }

    /**
     * Calls neural net, unless already have result cached, and returns a policy for the current player at given state.
     */
    shared_ptr<GoActionPolicy> GoEnv::get_policy_from_nn(shared_ptr<const GoState> state) {
        if (!state->nn_output_cache->has_cached_nn_output) {
            fill_cached_values_with_nn_output(state);
        }
        shared_ptr<GoActionPolicy> policy = make_shared<GoActionPolicy>();
        shared_ptr<GoActionVector> actions = get_valid_actions(state);
        for (shared_ptr<const GoAction> action : *actions) {
            int xlen = state->nn_output_cache->nn_x_len;
            int ylen = state->nn_output_cache->nn_y_len;
            int pos = NNPos::locToPos(action->loc, init_board.x_size, xlen, ylen);
            policy->insert_or_assign(action, state->nn_output_cache->cached_policy[pos]);
        }

        // normalise policy
        double sum_weights = 0.0;
        for (pair<shared_ptr<const GoAction>,double> pr : *policy) {
            sum_weights += pr.second;
        }
        double num_actions = policy->size();
        for (pair<shared_ptr<const GoAction>,double> pr : *policy) {
            shared_ptr<const GoAction> action = pr.first;
            (*policy)[action] = (sum_weights > 0.0) ? pr.second / sum_weights : 1.0 / num_actions;
        }

        return policy;
    }

    shared_ptr<ActionPrior> GoEnv::get_policy_from_nn_itfc(shared_ptr<const State> state) {
        shared_ptr<const GoState> go_state = static_pointer_cast<const GoState>(state);
        shared_ptr<GoActionPolicy> go_action_policy = get_policy_from_nn(go_state);
        shared_ptr<ActionPrior> action_prior = make_shared<ActionPrior>();
        for (pair<shared_ptr<const GoAction>,double> pr : *go_action_policy) {
            shared_ptr<const Action> action = static_pointer_cast<const Action>(pr.first);
            double prob = pr.second;
            action_prior->insert_or_assign(action, prob);
        }
        return action_prior;
    }

    /**
     * Heuristic fn 
    */
    double go_heuristic_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env) {
        shared_ptr<GoEnv> go_env = static_pointer_cast<GoEnv>(env);
        return go_env->get_heuristic_val_from_nn_itfc(state);
    }

    /**
     * Prior fn
    */
    shared_ptr<ActionPrior> go_prior_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env) {
        shared_ptr<GoEnv> go_env = static_pointer_cast<GoEnv>(env);
        return go_env->get_policy_from_nn_itfc(state);
    }

    /**
     * Black win prob from neural net
    */
    double GoEnv::get_black_win_prob_from_nn(shared_ptr<const GoState> state) {
        if (!state->nn_output_cache->has_cached_nn_output) {
            fill_cached_values_with_nn_output(state);
        }
        return state->nn_output_cache->cached_white_lose_prob; 
    }
}



/**
 * Boilerplate defined functions. Copied from thts_env_template.h.
 */
namespace thts {
    shared_ptr<GoStateDistr> GoEnv::get_observation_distribution(
        shared_ptr<const GoAction> action, shared_ptr<const GoState> next_state) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc);
        shared_ptr<GoStateDistr> distr;
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const GoState> obsv = static_pointer_cast<const GoState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const GoState> GoEnv::sample_observation_distribution(
        shared_ptr<const GoAction> action, 
        shared_ptr<const GoState> next_state, 
        RandManager& rand_manager) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const GoState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> GoEnv::sample_context(shared_ptr<const GoState> state) const
    {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}



/**
 * Boilerplate ThtsEnv interface implementation. Copied from thts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace thts {
    
    shared_ptr<const State> GoEnv::get_initial_state_itfc() const {
        shared_ptr<const GoState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool GoEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> GoEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        shared_ptr<vector<shared_ptr<const GoAction>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const GoAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> GoEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        shared_ptr<const GoAction> action_itfc = static_pointer_cast<const GoAction>(action);
        shared_ptr<GoStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const GoState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> GoEnv::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        shared_ptr<const GoAction> action_itfc = static_pointer_cast<const GoAction>(action);
        shared_ptr<const GoState> obsv = sample_transition_distribution(state_itfc, action_itfc);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> GoEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<const GoAction> act_itfc = static_pointer_cast<const GoAction>(action);
        shared_ptr<const GoState> next_state_itfc = static_pointer_cast<const GoState>(next_state);
        shared_ptr<GoStateDistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr;
        for (pair<const shared_ptr<const GoState>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }     

    shared_ptr<const Observation> GoEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
        RandManager& rand_manager) const
    {
        shared_ptr<const GoAction> act_itfc = static_pointer_cast<const GoAction>(action);
        shared_ptr<const GoState> next_state_itfc = static_pointer_cast<const GoState>(next_state);
        shared_ptr<const GoState> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double GoEnv::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        shared_ptr<const GoAction> action_itfc = static_pointer_cast<const GoAction>(action);
        shared_ptr<const GoState> obsv_itfc = static_pointer_cast<const GoState>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> GoEnv::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const GoState> state_itfc = static_pointer_cast<const GoState>(state);
        shared_ptr<ThtsEnvContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}