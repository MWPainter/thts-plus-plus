







// ---------------------------------------------------------------------------------------------------------------------
// Deterministic envs
// ---------------------------------------------------------------------------------------------------------------------

// BOOKMARK: deterministic envs


// static const std::unordered_set<std::string> DET_ENVS = 
// {
//     ENV_ID_D_CHAIN_10,
//     ENV_ID_MOD_D_CHAIN_10,
//     ENV_ID_ENTROPY_TRAP_10,
//     ENV_ID_FROZEN_LAKE_NO_HOLE_DENSE,
//     ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_LEN,
//     ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED,
//     ENV_ID_FROZEN_LAKE_D_8x8,
//     ENV_ID_FROZEN_LAKE_S_8x8,
//     ENV_ID_FROZEN_LAKE_D_8x16,
//     ENV_ID_FROZEN_LAKE_S_8x16,
//     ENV_ID_FROZEN_LAKE_D_16x16,
//     ENV_ID_FROZEN_LAKE_S_16x16,
// };
























// ---------------------------------------------------------------------------------------------------------------------
// Max trial lengths
// ---------------------------------------------------------------------------------------------------------------------



// BOOKMARK: max trial lengths


// // env ids - max trial length
// static const std::unordered_map<std::string,int> ENV_ID_MAX_TRIAL_LEN = 
// {
//     {D_CHAIN_10_ENV_ID,         10000},
//     {MOD_D_CHAIN_10_ENV_ID,     10000},
//     {ENTROPY_TRAP_10_ENV_ID,    10000},
//     {ENTROPY_TRAP_15_ENV_ID,    10000},
//     {FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID,50},
//     {FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID,50},
//     {FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID,50},
//     {FROZEN_LAKE_D_8x8_ENV_ID,    100},
//     {FROZEN_LAKE_S_8x8_ENV_ID,    100},
//     {FROZEN_LAKE_D_8x16_ENV_ID,    100},
//     {FROZEN_LAKE_S_8x16_ENV_ID,    100},
//     {FROZEN_LAKE_D_16x16_ENV_ID,    100},
//     {FROZEN_LAKE_S_16x16_ENV_ID,    100},
//     {SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID,    50},
//     {SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID,    50},
//     {SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID,    50},
//     {SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID,    50},
//     {SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID,    50},
//     {SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID,    50},
//     {SAILING_ENV_NORTH_ID,      100},
//     {SAILING_ENV_SOUTH_EAST_ID, 100},
//     {SAILING_8x16_ENV_NORTH_ID,      100},
//     {SAILING_8x16_ENV_SOUTH_EAST_ID, 100},
//     {SAILING_16x16_ENV_NORTH_ID,      100},
//     {SAILING_16x16_ENV_SOUTH_EAST_ID, 100},
// };









// ---------------------------------------------------------------------------------------------------------------------
// xpr ids
// ---------------------------------------------------------------------------------------------------------------------

// BOOKMARK: xpr ids



// // expr ids - debug
// static const std::string DEBUG_EXPR_ID = "000_debug";

// // expr ids - supp experiments (1xx + 2xx + 3xx)
// static const std::string SUPP_110_UCT_ON_FL_DENSE = "110_uct_on_fl_dense";
// static const std::string SUPP_111_UCT_ON_FL_SPARSE_LEN = "111_uct_on_fl_sparse_len";
// static const std::string SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED = "112_uct_on_fl_sparse_discounted";

// // expr ids - toy experiments (4xx + 5xx)
// // toy experiments = running experiments on the toy envs
// static const std::string TOY_XXX_EXPR_ID = "400_xxx";

// static const std::string EVAL_FL_S_8x8_EXPR_ID = "810_eval_fl_s_8x8";
// static const std::string EVAL_FL_S_8x16_EXPR_ID = "811_eval_fl_s_8x16";
// static const std::string EVAL_FL_S_16x16_EXPR_ID = "812_eval_fl_s_16x16";

// static const std::string EVAL_SFL_D_4x4_EXPR_ID = "820_eval_sfl_d_4x4";
// static const std::string EVAL_SFL_D_5x5_EXPR_ID = "821_eval_sfl_d_5x5";
// static const std::string EVAL_SFL_D_6x6_EXPR_ID = "822_eval_sfl_d_6x6";

// static const std::string EVAL_SFL_S_4x4_EXPR_ID = "830_eval_sfl_s_4x4";
// static const std::string EVAL_SFL_S_5x5_EXPR_ID = "831_eval_sfl_s_5x5";
// static const std::string EVAL_SFL_S_6x6_EXPR_ID = "832_eval_sfl_s_6x6";

// static const std::string EVAL_SAIL_N_8x8_EXPR_ID = "840_eval_sailing_n_8x8";
// static const std::string EVAL_SAIL_N_8x16_EXPR_ID = "841_eval_sailing_n_8x16";
// static const std::string EVAL_SAIL_N_16x16_EXPR_ID = "842_eval_sailing_n_16x16";

// static const std::string EVAL_SAIL_SE_8x8_EXPR_ID = "850_eval_sailing_se_8x8";
// static const std::string EVAL_SAIL_SE_8x16_EXPR_ID = "851_eval_sailing_se_8x16";
// static const std::string EVAL_SAIL_SE_16x16_EXPR_ID = "852_eval_sailing_se_16x16";



































// ---------------------------------------------------------------------------------------------------------------------
// old config written in a function :)
// ---------------------------------------------------------------------------------------------------------------------

// BOOKMARK: old xpr config 






//     /**
//      * Gets a list of RunID objects from a given expr id
//     */
//     shared_ptr<vector<RunID>> get_run_ids_from_expr_id_prefix(string expr_id_prefix) 
//     {   
//         string expr_id = lookup_expr_id_from_prefix(expr_id_prefix);
//         shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();





//         // ----
//         // expr_id: 110_uct_on_fl_dense / 111_uct_on_fl_sparse_len / 112_uct_on_fl_sparse_discounted 
//         // sanity check UCT on frozen lake stuff??
//         // ----
//         if (expr_id == SUPP_110_UCT_ON_FL_DENSE
//             || expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN
//             || expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) 
//         {
//             double default_q_value = -50;
//             string env_id = FROZEN_LAKE_NO_HOLE_DENSE_ENV_ID;
//             if (expr_id == SUPP_111_UCT_ON_FL_SPARSE_LEN) {
//                 env_id = FROZEN_LAKE_NO_HOLE_SPARSE_LEN_ENV_ID;
//             } else if (expr_id == SUPP_112_UCT_ON_FL_SPARSE_DISCOUNTED) {
//                 env_id = FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED_ENV_ID;
//                 default_q_value = 0;
//             }
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 5000;
//             double eval_delta = 50;
//             int rollouts_per_mc_eval = 1; // det env
//             int num_repeats = 25;
//             int num_threads = 8;
//             int eval_threads = 1; // det env
            
//             // UCT run ids 
//             vector<double> biases_to_try = {
//                 // UctManagerArgs::bias_default,
//                 0.001,
//                 0.01,
//                 0.1,
//                 0.18,
//                 0.32,
//                 0.58,
//                 1.0,
//                 1.8,
//                 3.2,
//                 5.8,
//                 10.0,
//                 18.0,
//                 32.0,
//                 58.0,
//                 100.0,
//                 1000.0,
//                 10000.0,
//             };

//             for (double bias : biases_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {BIAS_PARAM_ID, bias},
//                 };
//                 run_ids->push_back(RunID(
//                     env_id,
//                     expr_id,
//                     expr_timestamp,
//                     UCT_ALG_ID,
//                     alg_params,
//                     eval_wrt_time,
//                     search_runtime,
//                     eval_delta,
//                     rollouts_per_mc_eval,
//                     max_trial_length,
//                     num_repeats,
//                     num_threads,
//                     eval_threads
//                 ));
//             }
            
//             // MENTS/DENTS/BTS run ids 
//             vector<double> temps_to_try = {
//                 0.001,
//                 0.01,
//                 0.018,
//                 0.032,
//                 0.058,
//                 0.1,
//                 0.18,
//                 0.32,
//                 0.58,
//                 1.0,
//                 1.8,
//                 3.2,
//                 5.8,
//                 10.0,
//                 18.0,
//                 32.0,
//                 58.0,
//                 100.0,
//                 1000.0,
//                 10000.0,
//             };
//             vector<string> alg_ids = 
//             {
//                 MENTS_ALG_ID,
//                 BTS_ALG_ID,
//                 DENTS_ALG_ID,
//             };

//             for (double temp : temps_to_try) {
//                 unordered_map<string,double> alg_params =
//                 {
//                     {TEMP_PARAM_ID, temp},
//                     {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {ENTROPY_COEFF_PARAM_ID, temp},
//                     {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                     {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 1.0},
//                     {EPSILON_PARAM_ID, 0.01},
//                     {DEFAULT_Q_VALUE_PARAM_ID, default_q_value}
//                 };
//                 for (string alg_id : alg_ids) {
//                     run_ids->push_back(RunID(
//                         env_id,
//                         expr_id,
//                         expr_timestamp,
//                         alg_id,
//                         alg_params,
//                         eval_wrt_time,
//                         search_runtime,
//                         eval_delta,
//                         rollouts_per_mc_eval,
//                         max_trial_length,
//                         num_repeats,
//                         num_threads,
//                         eval_threads
//                     ));
//                 }
//             }
            
//             return run_ids;
//         }




//         // ----
//         // expr_id: 82x - slippy frozen lake, stochastic, dense reward
//         // ----
//         if (expr_id == EVAL_SFL_D_4x4_EXPR_ID || expr_id == EVAL_SFL_D_5x5_EXPR_ID || expr_id == EVAL_SFL_D_6x6_EXPR_ID)
//         {
//             // Env params
//             string env_id = SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID;
//             if (expr_id == EVAL_SFL_D_5x5_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_D_5x5_ENV_ID;
//             } else if (expr_id == EVAL_SFL_D_6x6_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_D_6x6_ENV_ID;
//             }
//             double default_q_value = - (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // -23.5199
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.0545607},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // -23.5268
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 2.74005},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // -23.4768
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.859055},
//                 {UCT_BUDGET_PARAM_ID, 3},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // -23.5393
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // -23.5238
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.036216},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 99.4302},
//                 {EPSILON_PARAM_ID, 0.996121},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // -23.4732
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 100},
//                 {ENTROPY_COEFF_PARAM_ID, 0.00547603},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 0.01},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // -23.559
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 1.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // -23.5146
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.00829662},
//                 {EPSILON_PARAM_ID, 0.00019767},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }




//         // ----
//         // expr_id: 83x - slippy frozen lake, stochastic, sparse reward
//         // ----
//         if (expr_id == EVAL_SFL_S_4x4_EXPR_ID || expr_id == EVAL_SFL_S_5x5_EXPR_ID || expr_id == EVAL_SFL_S_6x6_EXPR_ID)
//         {
//             // Env params
//             string env_id = SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID;
//             if (expr_id == EVAL_SFL_S_5x5_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_S_5x5_ENV_ID;
//             } else if (expr_id == EVAL_SFL_S_6x6_EXPR_ID) {
//                 env_id = SLIPPY_FROZEN_LAKE_S_6x6_ENV_ID;
//             }
//             double default_q_value = 0.0;
//             int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//             time_t expr_timestamp = std::time(nullptr);
//             bool eval_wrt_time = false;
//             double search_runtime = 250000;
//             double eval_delta = 500;
//             int rollouts_per_mc_eval = 1024;
//             int num_repeats = 25;
//             int num_threads = 16;
//             int eval_threads = 16;

//             string alg_id;
//             unordered_map<string,double> alg_params;

//             // UCT 
//             // 0.0382813
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.104752},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MaxUCT 
//             // 0.0392578
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.247274},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // HMCTS
//             // 0.0386719
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 0.0988045},
//                 {UCT_BUDGET_PARAM_ID, 517},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // MENTS
//             // 0.0410156
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // BTS
//             // 0.0404297
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.00414547},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0101548},
//                 {EPSILON_PARAM_ID, 0.227683},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // DENTS
//             // 0.0396484
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0738441},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 42.4924},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0826038},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 28.191},
//                 {EPSILON_PARAM_ID, 0.710385},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // RENTS
//             // 0.0380859
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             // TENTS
//             // 0.040625
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.001},
//                 {EPSILON_PARAM_ID, 0.0},
//                 {DEFAULT_Q_VALUE_PARAM_ID, default_q_value},
//             };
//             run_ids->push_back(RunID(
//                 env_id,
//                 expr_id,
//                 expr_timestamp,
//                 alg_id,
//                 alg_params,
//                 eval_wrt_time,
//                 search_runtime,
//                 eval_delta,
//                 rollouts_per_mc_eval,
//                 max_trial_length,
//                 num_repeats,
//                 num_threads,
//                 eval_threads
//             ));

//             return run_ids;
//         }