#include "go/run_go.h"

#include "mc_eval.h"
#include "thts.h"
#include "helper_templates.h"

#include "algorithms/uct/alphago_manager.h"
#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/puct_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "algorithms/uct/alphago_decision_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
#include "algorithms/est/est_decision_node.h"

#include "go/go_env.h"
#include "go/go_state_action.h"

#include "KataGo/cpp/search/search.h"
#include "KataGo/cpp/search/searchparams.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace std;

namespace thts {
    /**
     * Gets the results directory for this run (doesn't check/make)
    */
    string get_results_dir(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        bool use_filenames_for_hps, 
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key)
    {
        stringstream ss;
        ss << "results/go/" 
            << expr_id << "/";
        if (!use_filenames_for_hps) {
            ss << alg1_id << "_vs_" << alg2_id << "(size=" << board_size << ",komi=" << komi << ")/";
        } else {
            string alg1_str = (hps_key == "") ? alg1_id : to_string(alg_params->at(hps_key));
            string alg2_str = (hps_opp_key == "") ? alg2_id : to_string(alg_params->at(hps_opp_key));
            ss << alg1_str << "_vs_" << alg2_str << "/";
        }
        return ss.str();
    }

    /**
     * Returns the results directory to use for this run, making sure that it exists, and creating it if it doesnt
    */
    string create_results_dir(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        bool use_filenames_for_hps, 
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key)
    {
        string results_dir = get_results_dir(
            expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        if (!filesystem::exists(results_dir)) {
            filesystem::create_directories(results_dir);
        }
        return results_dir;
    }

    /**
     * Returns the filename for game results
    */
    string get_results_filename(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        bool use_filenames_for_hps, 
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key)
    {
        stringstream ss;
        ss << get_results_dir(
            expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        ss << "results.csv";
        return ss.str();
    }

    /**
     * Writes the header for results file
    */
    void write_results_file_header(
        ofstream& results_file, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        double time_per_move, 
        int num_threads) 
    {
        results_file << "Results for match " 
            << alg1_id << " (black) vs " << alg2_id << " (white), "
            << "with board size " << board_size << " and komi of " << komi
            << ". Algorithms given " << time_per_move << "s to move, and can use " << num_threads << " threads in thts."
            << endl;
        results_file << "#match,result,score,cumulative_black_wins,cumulative_white_wins" << endl;
    }

    /**
     * Write a match result
    */
    void write_match_result_in_results_file(
        ofstream& results_file, int game_num, double result, double score, int black_wins, int white_wins) 
    {
        results_file << game_num << ","
            << result << ","
            << score << ","
            << black_wins << ","
            << white_wins << endl;
    }

    /**
     * Returns the filename forva match
    */
    string get_match_filename(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        int game,
        bool use_filenames_for_hps, 
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key)
    {
        stringstream ss;
        ss << get_results_dir(expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        ss << "match_" << game << ".csv";
        return ss.str();
    }

    /**
     * Writes the header for match file
    */
    string write_match_file_header(
        ofstream& match_file, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        double time_per_move, 
        int num_threads,
        int game,
        double init_nn_eval, 
        double init_nn_black_win) 
    {
        stringstream ss;
        ss << "Details for the " << game << "th match of " << alg1_id << " (black) vs " << alg2_id << " (white), "
            << "with board size " << board_size << " and komi of " << komi << ". Algorithms given " << time_per_move 
            << "s to move, and can use " << num_threads 
            << " threads in thts. For reference, the initial nn_eval and nn_black_win_prob are initially " 
            << init_nn_eval << " and " << init_nn_black_win << " respectively." << endl;
        ss << "#move,action,pos,alg,num_trials,nn_eval,nn_black_win_prob" << endl;
        string match_file_header = ss.str();
        match_file << match_file_header;
        match_file.flush();
        return match_file_header;
    }
    
    /**
     * Write a move for a match log
    */
    string write_move_in_match_file(
        ofstream& match_file, 
        int move_counter, 
        shared_ptr<const GoAction> go_action, 
        shared_ptr<ThtsDNode> root_node,
        int board_size,
        string alg_id,
        double nn_eval,
        double nn_black_win,
        int num_visits=0) 
    {
        stringstream ss;
        ss << move_counter << ","
            << go_action->loc << ","
            << "(" << go_action->get_x_coord(board_size) << "|" << go_action->get_y_coord(board_size) << "),"
            << alg_id << ","
            << ((root_node != nullptr) ? root_node->get_num_visits() : num_visits) << "," 
            << nn_eval << "," 
            << nn_black_win << endl;
        string match_line = ss.str();
        match_file << match_line;
        match_file.flush();
        return match_line;
    }

    /**
     * Returns the filename for tree print out
    */
    string get_tree_print_filename(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        double komi, 
        bool use_filenames_for_hps, 
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key,
        int match_num,
        int move_num) 
    {
        stringstream ss_dir;
        ss_dir << get_results_dir(
            expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        ss_dir << "trees/match_" << match_num << "/";
        string tree_dir = ss_dir.str();
        if (!filesystem::exists(tree_dir)) {
            filesystem::create_directories(tree_dir);
        }

        stringstream ss;
        ss << tree_dir;
        if ((move_num & 1) == 0) {
            ss << alg1_id;
        } else {
            ss << alg2_id;
        }
        ss << "_" << move_num;
        return ss.str();
    }

    /**
     * Print tree to file for debug
    */
    void print_tree_to_file(ofstream& tree_file, shared_ptr<ThtsDNode> node, int depth=2) {
        tree_file << node->get_pretty_print_string(depth) << endl;
    }

    /**
     * Helper to check if key
    */
    bool contains_key(shared_ptr<GoAlgParams> alg_params, string key) {
        return alg_params->find(key) != alg_params->end();
    }

    /**
     * Make manager for thts (and sets params to use for the search)
    */
    shared_ptr<ThtsManager> make_manager(
        shared_ptr<GoEnv> go_env, 
        shared_ptr<const State> init_state, 
        string algo_id_for_this_move, 
        int board_size, 
        shared_ptr<GoAlgParams> alg_params,
        bool is_opp) 
    {
        if (algo_id_for_this_move == ALG_ID_KATA) 
        {
            AlphaGoManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_most_visited = true;
            manager_args.heuristic_psuedo_trials = 1.0;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            manager_args.dirichlet_noise_coeff = 0.0;
            manager_args.dirichlet_noise_param = 0.0;
            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                    if (contains_key(alg_params, PARAM_KATA_RECOMMEND_AVG_RETURN)) {
                        manager_args.recommend_most_visited = false;
                    }
                    if (contains_key(alg_params, PARAM_USE_DIRICHLET_NOISE)) {
                        int num_moves_avail = go_env->get_valid_actions_itfc(init_state)->size();
                        manager_args.dirichlet_noise_coeff = 0.25;
                        manager_args.dirichlet_noise_param = 0.03 * board_size * board_size / num_moves_avail;
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                    if (contains_key(alg_params, PARAM_KATA_RECOMMEND_AVG_RETURN_OPP)) {
                        manager_args.recommend_most_visited = false;
                    }
                    if (contains_key(alg_params, PARAM_USE_DIRICHLET_NOISE_OPP)) {
                        int num_moves_avail = go_env->get_valid_actions_itfc(init_state)->size();
                        manager_args.dirichlet_noise_coeff = 0.25;
                        manager_args.dirichlet_noise_param = 0.03 * board_size * board_size / num_moves_avail;
                    }
                }
            }
            return make_shared<AlphaGoManager>(manager_args);
        }

        if (algo_id_for_this_move == ALG_ID_UCT) 
        {
            UctManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_most_visited = true;
            manager_args.heuristic_psuedo_trials = 1.0;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                }
            }
            return make_shared<UctManager>(manager_args);
        } 

        if (algo_id_for_this_move == ALG_ID_PUCT) 
        {
            PuctManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_most_visited = true;
            manager_args.heuristic_psuedo_trials = 1.0;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                }
            }
            return make_shared<PuctManager>(manager_args);
        }

        if (algo_id_for_this_move == ALG_ID_UNI) {
            UctManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_most_visited = false;
            manager_args.heuristic_psuedo_trials = 1.0;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            manager_args.epsilon_exploration = 1.0;
            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.bias = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                }
            }
            return make_shared<UctManager>(manager_args);
        }

        if (algo_id_for_this_move == ALG_ID_MENTS 
            || algo_id_for_this_move == ALG_ID_RENTS 
            || algo_id_for_this_move == ALG_ID_TENTS) 
        {
            MentsManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_visit_threshold = 20;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            manager_args.epsilon = 0.03;
            manager_args.root_node_epsilon = 0.7;
            manager_args.shift_pseudo_q_values = true;
            manager_args.prior_policy_search_weight = 0.5;
            // manager_args.use_max_heap = (algo_id_for_this_move != ALG_ID_TENTS);
            manager_args.avoid_selecting_children_under_construction = true;

            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF);
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_avg_return = true;
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS_OPP)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS_OPP)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF_OPP)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF_OPP);
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN_OPP)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_avg_return = true;
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED_OPP)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS_OPP)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                }
            }
            return make_shared<MentsManager>(manager_args);
        }
        if (algo_id_for_this_move == ALG_ID_DENTS) 
        {
            DentsManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_visit_threshold = 20;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            manager_args.epsilon = 0.03;
            manager_args.root_node_epsilon = 0.67;
            manager_args.shift_pseudo_q_values = true;
            manager_args.prior_policy_search_weight = 0.5;
            manager_args.value_temp_decay_fn = decayed_temp_inv_log; // decayed_temp_inv_sqrt;
            manager_args.use_max_heap = true;
            manager_args.avoid_selecting_children_under_construction = true;
            
            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                        if (!contains_key(alg_params, PARAM_INIT_DECAY_TEMP)) {
                            manager_args.value_temp_init = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                        }
                    }
                    if (contains_key(alg_params, PARAM_INIT_DECAY_TEMP)) {
                        manager_args.value_temp_init = alg_params->at(PARAM_INIT_DECAY_TEMP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE)) {
                        manager_args.value_temp_decay_root_node_visits_scale = alg_params->at(
                            PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_VISITS_SCALE)) {
                        manager_args.value_temp_decay_visits_scale = alg_params->at(PARAM_DECAY_TEMP_VISITS_SCALE);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_USE_SIGMOID)) {
                        manager_args.value_temp_decay_fn = decayed_temp_sigmoid;
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_dp_value = false;
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_CONST_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_SQRT_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_LOG_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_log;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                        if (!contains_key(alg_params, PARAM_INIT_DECAY_TEMP_OPP)) {
                            manager_args.value_temp_init = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                        }
                    }
                    if (contains_key(alg_params, PARAM_INIT_DECAY_TEMP_OPP)) {
                        manager_args.value_temp_init = alg_params->at(PARAM_INIT_DECAY_TEMP_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS_OPP)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS_OPP)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF_OPP)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF_OPP);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP)) {
                        manager_args.value_temp_decay_root_node_visits_scale = alg_params->at(
                            PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_VISITS_SCALE_OPP)) {
                        manager_args.value_temp_decay_visits_scale = alg_params->at(PARAM_DECAY_TEMP_VISITS_SCALE_OPP);
                    }
                    if (contains_key(alg_params, PARAM_DECAY_TEMP_USE_SIGMOID_OPP)) {
                        manager_args.value_temp_decay_fn = decayed_temp_sigmoid;
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN_OPP)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_dp_value = false;
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_CONST_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_LOG_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_log;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED_OPP)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS_OPP)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                }
            }
            return make_shared<DentsManager>(manager_args);
        }
        if (algo_id_for_this_move == ALG_ID_EST) 
        {
            DentsManagerArgs manager_args(go_env);
            manager_args.is_two_player_game = true;
            manager_args.mcts_mode = true;
            manager_args.recommend_visit_threshold = 20;
            manager_args.heuristic_fn = go_heuristic_fn;
            manager_args.prior_fn = go_prior_fn;
            manager_args.epsilon = 0.03;
            manager_args.root_node_epsilon = 0.67;
            manager_args.shift_pseudo_q_values = true;
            manager_args.prior_policy_search_weight = 0.5;
            manager_args.value_temp_init = 0.0;
            // manager_args.temp_decay_visits_scale = 15.0;
            // manager_args.temp_decay_root_node_visits_scale = 15.0;
            manager_args.use_max_heap = true;
            manager_args.avoid_selecting_children_under_construction = true;

            if (alg_params != nullptr) {
                if (!is_opp) {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF);
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_dp_value = false;
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_CONST_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_SQRT_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_LOG_SEARCH_TEMP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_log;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                } else {
                    if (contains_key(alg_params, PARAM_BIAS_OR_SEARCH_TEMP_OPP)) {
                        manager_args.temp = alg_params->at(PARAM_BIAS_OR_SEARCH_TEMP_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_ROOT_EPS_OPP)) {
                        manager_args.root_node_epsilon = alg_params->at(PARAM_MENTS_ROOT_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_MENTS_EPS_OPP)) {
                        manager_args.epsilon = alg_params->at(PARAM_MENTS_EPS_OPP);
                    }
                    if (contains_key(alg_params, PARAM_PRIOR_COEFF_OPP)) {
                        manager_args.prior_policy_search_weight = alg_params->at(PARAM_PRIOR_COEFF_OPP);
                    }
                    if (contains_key(alg_params, PARAM_USE_AVG_RETURN_OPP)) {
                        manager_args.use_max_heap = false;
                        manager_args.use_dp_value = false;
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_CONST_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_no_decay;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_sqrt;
                    }
                    if (contains_key(alg_params, PARAM_USE_INV_LOG_SEARCH_TEMP_OPP)) {
                        manager_args.temp_decay_fn = decayed_temp_inv_log;
                    }
                    if (contains_key(alg_params, PARAM_RECOMMEND_MOST_VISITED_OPP)) {
                        manager_args.recommend_most_visited = true;
                    }
                    if (contains_key(alg_params, PARAM_USE_ALIAS_METHODS_OPP)) {
                        manager_args.alias_use_caching = true;
                        manager_args.alias_recompute_freq = 1;
                    }
                }
            }
            
            return make_shared<DentsManager>(manager_args);
        }

        throw runtime_error("Error in making thts_manager");
    }
    
    /**
     * Make the root node for a search
    */
    shared_ptr<ThtsDNode> make_root_node(
        shared_ptr<ThtsEnv> go_env, 
        shared_ptr<ThtsManager> manager, 
        shared_ptr<const State> cur_state,
        string algo_id_for_this_move, 
        int move_counter) 
    {
        if (algo_id_for_this_move == ALG_ID_KATA) {
            shared_ptr<AlphaGoManager> alpha_manager = static_pointer_cast<AlphaGoManager>(manager);
            return make_shared<AlphaGoDNode>(alpha_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_PUCT) {
            shared_ptr<PuctManager> puct_manager = static_pointer_cast<PuctManager>(manager);
            return make_shared<PuctDNode>(puct_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_UNI) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_MENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<MentsDNode>(ments_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_RENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<RentsDNode>(ments_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_TENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<TentsDNode>(ments_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_DENTS) {
            shared_ptr<DentsManager> dents_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<DentsDNode>(dents_manager, cur_state, 0, move_counter);
        }
        if (algo_id_for_this_move == ALG_ID_EST) {
            shared_ptr<DentsManager> ments_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<EstDNode>(ments_manager, cur_state, 0, move_counter);
        }

        throw runtime_error("Error in make_root_node");

    }

    /**
     * Running KataGo directly
    */
    shared_ptr<const GoAction> get_katago_move(
        GoEnv& env, const GoState& state, bool is_white, double max_time, int max_trials, int& num_playouts, bool use_bts) 
    {
        // Setup search params, took values from match config given in: https://github.com/lightvector/KataGo/blob/master/cpp/configs/match_example.cfg
        // If not found in the config, values were kept at default
        // Except: used the quotes cpuct = 1.1 from the paper
        SearchParams params;
        //Utility function parameters
        params.winLossUtilityFactor = 1.0;
        params.staticScoreUtilityFactor = 0.1;
        params.dynamicScoreUtilityFactor = 0.3;
        params.dynamicScoreCenterZeroWeight = 0.2;
        params.dynamicScoreCenterScale = 0.75; 
        params.noResultUtilityForWhite = 0.0; 
        params.drawEquivalentWinsForWhite = 0.5; 

        //Search tree exploration parameters
        params.cpuctExploration = 1.1;
        params.cpuctExplorationLog = 0.4;
        params.fpuReductionMax = 0.2;
        params.fpuParentWeightByVisitedPolicy = true;

        //Tree value aggregation parameters
        params.valueWeightExponent = 0.25;

        //Root parameters
        params.rootNumSymmetriesToSample = 1;
        params.rootFpuReductionMax = 0.1;

        //Parameters for choosing the move to play
        params.chosenMoveTemperature = 0.2;
        params.chosenMoveTemperatureEarly = 0.6;
        params.chosenMoveTemperatureHalflife = 19;
        params.chosenMoveSubtract = 0.0;
        params.chosenMovePrune = 1.0;
        params.useLcbForSelection = true;
        params.lcbStdevs = 5.0;
        params.minVisitPropForLCB = 0.15;

        //Mild behavior hackery
        params.rootEndingBonusPoints = 0.5;
        params.rootPruneUselessMoves = true;
        params.subtreeValueBiasFactor = 0.45;
        params.subtreeValueBiasWeightExponent = 0.85;

        //Threading-related
        params.nodeTableShardsPowerOfTwo = 16;
        params.numVirtualLossesPerThread = 1.0;
        params.numThreads = 32;
        params.maxVisits = ((int64_t)1) << 50;
        params.maxPlayouts = max_trials;

        // TIME CONTROL
        params.maxTime = max_time;

        // No cheating - only search when for the 'max_time' allowed
        params.maxVisitsPondering = 0;
        params.maxPlayoutsPondering = 0;
        params.maxTimePondering = 0.0;
        
        stringstream ss;
        ss << "60415";
        if (is_white) {
            ss << "@W";
        } else {
            ss << "@B";
        }
        string seed = ss.str();
        Player pla = (is_white) ? P_WHITE : P_BLACK; 

        Search katago_search(params, env.get_nn_eval(), env.get_logger(), seed, use_bts);
        katago_search.setPosition(pla, state.get_current_board(), *state.get_board_history()); // pass in these args
        Loc loc = katago_search.runWholeSearchAndGetMove(pla);

        num_playouts = katago_search.lastSearchNumPlayouts;

        return make_shared<const GoAction>(loc);
    }

    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
     * (This is the one exposed function (for now) in run_toy.cpp)
    */
    void run_go_games(
        string expr_id, 
        string alg1_id, 
        string alg2_id, 
        int board_size, 
        int num_games, 
        double komi, 
        bool use_time_controls,
        double trials_or_time_per_move,
        int num_threads,
        bool use_filenames_for_hps,
        shared_ptr<GoAlgParams> alg_params,
        string hps_key,
        string hps_opp_key)
    {
        
        // create results dir + results file
        create_results_dir(
            expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        string results_filename = get_results_filename(
            expr_id, alg1_id, alg2_id, board_size, komi, use_filenames_for_hps, alg_params, hps_key, hps_opp_key);
        ofstream results_file;
        results_file.open(results_filename, ios::out);// | ios::app);
        write_results_file_header(
            results_file, alg1_id, alg2_id, board_size, komi, trials_or_time_per_move, num_threads);

        // vars counting wins
        int black_wins = 0;
        int white_wins = 0;

        // Early stopping variables (hard coded)
        double early_stop_win_prob_threshold = 0.001;
        int early_stop_moves_req = 10;
        int full_playout_freq = 1;


        for (int game=0; game<num_games; game++) {
            // print
            cout << "Starting game number " << game << " between " << alg1_id << " and " << alg2_id << " with komi " 
                << komi << endl;

            // make env
            shared_ptr<GoEnv> go_env = make_shared<GoEnv>(board_size, komi);
            shared_ptr<const GoState> cur_state = go_env->get_initial_state();
            go_env->update_dynamic_score_center_for_root_state(cur_state);
                
            // Get the neural net eval for the initial state
            double init_nn_eval = go_env->get_heuristic_val_from_nn(cur_state);
            double init_nn_black_win = go_env->get_black_win_prob_from_nn(cur_state);
            
            // Make match file
            string match_filename = get_match_filename(
                expr_id, 
                alg1_id, 
                alg2_id, 
                board_size, 
                komi, 
                game, 
                use_filenames_for_hps, 
                alg_params, 
                hps_key, 
                hps_opp_key);
            ofstream match_file;
            match_file.open(match_filename, ios::out);// | ios::app);
            string match_file_header = write_match_file_header(
                match_file, 
                alg1_id, 
                alg2_id, 
                board_size, 
                komi, 
                trials_or_time_per_move, 
                num_threads, 
                game, 
                init_nn_eval, 
                init_nn_black_win);
            cout << match_file_header;
            cout.flush();

            // vars updating through loops
            string algo_ids[] = {alg1_id, alg2_id};
            int i = 0;
            int move_counter = 0;
            bool full_playout = (game % full_playout_freq) == 0;
            bool early_stop = false;
            int early_stop_moves_black_winning = 0;
            int early_stop_moves_white_winning = 0;

            while (!go_env->is_sink_state_itfc(cur_state) && !early_stop) {

                // get next player + update i for next move
                string algo_id_for_this_move = algo_ids[i];
                bool is_opp = (i != 0);
                i = 1-i;

                // Forward declrs
                shared_ptr<const GoAction> cur_action;
                shared_ptr<ThtsDNode> root_node;
                int num_playouts = 0;

                // Get move from katago
                if (algo_id_for_this_move == ALG_ID_KATA_NATIVE || algo_id_for_this_move == ALG_ID_KATA_NATIVE_BTS) {
                    int trials_per_move = numeric_limits<int>::max();
                    double time_per_move = numeric_limits<double>::max();
                    if (use_time_controls) {
                        time_per_move = trials_or_time_per_move;
                    } else {
                        trials_per_move = (int) trials_or_time_per_move;
                    }
                    bool use_bts = (algo_id_for_this_move == ALG_ID_KATA_NATIVE_BTS);
                    cur_action = get_katago_move(
                        *go_env, *cur_state, is_opp, time_per_move, trials_per_move, num_playouts, use_bts);

                } else { 
                    // otherwise setup+run thts for this move
                    int num_threads_for_this_move = num_threads;
                    if (!is_opp && contains_key(alg_params, NUM_THREADS_OVERRIDE)) {
                        num_threads_for_this_move = alg_params->at(NUM_THREADS_OVERRIDE);
                    } else if (is_opp && contains_key(alg_params, NUM_THREADS_OVERRIDE_OPP)) {
                        num_threads_for_this_move = alg_params->at(NUM_THREADS_OVERRIDE_OPP);
                    }
                    shared_ptr<ThtsManager> thts_manager = make_manager(
                        go_env, cur_state, algo_id_for_this_move, board_size, alg_params, is_opp);
                    root_node = make_root_node(
                        go_env, thts_manager, cur_state, algo_id_for_this_move, move_counter);
                    ThtsPool thts_pool(thts_manager, root_node, num_threads_for_this_move);
                    int trials_per_move = numeric_limits<int>::max();
                    double time_per_move = numeric_limits<double>::max();
                    if (use_time_controls) {
                        time_per_move = trials_or_time_per_move;
                    } else {
                        trials_per_move = (int) trials_or_time_per_move;
                    }
                    thts_pool.run_trials(trials_per_move, time_per_move);

                    // Get action recommended by thts/
                    cur_action = static_pointer_cast<const GoAction>(
                        root_node->recommend_action_itfc(*go_env->sample_context(cur_state)));
                }
                
                // Perform move recommended by thts/katago
                cur_state = go_env->sample_transition_distribution(cur_state, cur_action);
                // Get the neural net eval for the current state
                double nn_eval = go_env->get_heuristic_val_from_nn(cur_state);
                double nn_black_win = go_env->get_black_win_prob_from_nn(cur_state);
                go_env->update_dynamic_score_center_for_root_state(cur_state);

                // Log move in match
                string move_string = write_move_in_match_file(
                    match_file, 
                    move_counter, 
                    cur_action, 
                    root_node, 
                    board_size, 
                    algo_id_for_this_move, 
                    nn_eval, 
                    nn_black_win,
                    num_playouts);
                cout << move_string;
                cout.flush();

                // Print tree to file (if not katago native)
                if (algo_id_for_this_move != ALG_ID_KATA_NATIVE && algo_id_for_this_move != ALG_ID_KATA_NATIVE_BTS) {
                    string tree_filename = get_tree_print_filename(
                        expr_id, 
                        alg1_id, 
                        alg2_id, 
                        board_size, 
                        komi, 
                        use_filenames_for_hps, 
                        alg_params, 
                        hps_key, 
                        hps_opp_key, 
                        game, 
                        move_counter);
                    ofstream tree_file;
                    tree_file.open(tree_filename, ios::out);
                    print_tree_to_file(tree_file, root_node);
                    tree_file.close();
                }

                // increment move
                move_counter++;

                // check if early stop
                if (!full_playout) {
                    if (nn_black_win <= early_stop_win_prob_threshold) {
                        early_stop_moves_black_winning = 0;
                        early_stop_moves_white_winning++;
                    } else if (1.0 - nn_black_win <= early_stop_win_prob_threshold) {
                        early_stop_moves_black_winning++;
                        early_stop_moves_white_winning = 0;
                    } else {
                        early_stop_moves_black_winning = 0;
                        early_stop_moves_white_winning = 0;
                    }

                    early_stop = (early_stop_moves_black_winning >= early_stop_moves_req);
                    early_stop = early_stop || (early_stop_moves_white_winning >= early_stop_moves_req);
                }
            }

            // get result and score of the match
            // n.b. score will be 0.0 if early stop (ends with a .5 from komi otherwise)
            double result = cur_state->get_result();
            double score = cur_state->get_score();
            if (early_stop) {
                if (early_stop_moves_black_winning > 0) {
                    result = 1.0;
                } else if (early_stop_moves_white_winning > 0) {
                    result = -1.0;
                } else {
                    throw runtime_error("Shouldn't be early stopping without a clear winner");
                }
                score = 0.0;
            }
            if (result > 0.0) black_wins++;
            if (result < 0.0) white_wins++;

            // write match result in results file
            write_match_result_in_results_file(results_file, game, result, score, black_wins, white_wins);

            // close match file
            match_file.close();
        }

        // close results file
        results_file.close();
    }
}
