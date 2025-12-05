







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

// // expr ids - rerun experiments (6xx + 7xx = hyperparam, 8xx + 9xx = eval)
// // rerunning experiments = repeating the experiments with hyperparam tuning now
// static const std::string HP_OPT_600_UCT_EXPR_ID =       "600_hp_opt_uct";
// static const std::string HP_OPT_610_MAX_UCT_EXPR_ID =   "610_hp_opt_max_uct";
// static const std::string HP_OPT_620_MENTS_EXPR_ID =     "620_hp_opt_ments";
// static const std::string HP_OPT_630_BTS_EXPR_ID =       "630_hp_opt_bts";
// static const std::string HP_OPT_640_DENTS_EXPR_ID =     "640_hp_opt_dents";
// static const std::string HP_OPT_650_RENTS_EXPR_ID =     "650_hp_opt_rents";
// static const std::string HP_OPT_660_TENTS_EXPR_ID =     "660_hp_opt_tents";
// static const std::string HP_OPT_670_HMCTS_EXPR_ID =     "670_hp_opt_hmcts";

// static const std::string HP_OPT_601_UCT_EXPR_ID =       "601_hp_opt_uct";
// static const std::string HP_OPT_611_MAX_UCT_EXPR_ID =   "611_hp_opt_max_uct";
// static const std::string HP_OPT_621_MENTS_EXPR_ID =     "621_hp_opt_ments";
// static const std::string HP_OPT_631_BTS_EXPR_ID =       "631_hp_opt_bts";
// static const std::string HP_OPT_641_DENTS_EXPR_ID =     "641_hp_opt_dents";
// static const std::string HP_OPT_651_RENTS_EXPR_ID =     "651_hp_opt_rents";
// static const std::string HP_OPT_661_TENTS_EXPR_ID =     "661_hp_opt_tents";
// static const std::string HP_OPT_671_HMCTS_EXPR_ID =     "671_hp_opt_hmcts";

// static const std::string HP_OPT_602_UCT_EXPR_ID =       "602_hp_opt_uct";
// static const std::string HP_OPT_612_MAX_UCT_EXPR_ID =   "612_hp_opt_max_uct";
// static const std::string HP_OPT_622_MENTS_EXPR_ID =     "622_hp_opt_ments";
// static const std::string HP_OPT_632_BTS_EXPR_ID =       "632_hp_opt_bts";
// static const std::string HP_OPT_642_DENTS_EXPR_ID =     "642_hp_opt_dents";
// static const std::string HP_OPT_652_RENTS_EXPR_ID =     "652_hp_opt_rents";
// static const std::string HP_OPT_662_TENTS_EXPR_ID =     "662_hp_opt_tents";
// static const std::string HP_OPT_672_HMCTS_EXPR_ID =     "672_hp_opt_hmcts";

// static const std::string HP_OPT_603_UCT_EXPR_ID =       "603_hp_opt_uct";
// static const std::string HP_OPT_613_MAX_UCT_EXPR_ID =   "613_hp_opt_max_uct";
// static const std::string HP_OPT_623_MENTS_EXPR_ID =     "623_hp_opt_ments";
// static const std::string HP_OPT_633_BTS_EXPR_ID =       "633_hp_opt_bts";
// static const std::string HP_OPT_643_DENTS_EXPR_ID =     "643_hp_opt_dents";
// static const std::string HP_OPT_653_RENTS_EXPR_ID =     "653_hp_opt_rents";
// static const std::string HP_OPT_663_TENTS_EXPR_ID =     "663_hp_opt_tents";
// static const std::string HP_OPT_673_HMCTS_EXPR_ID =     "673_hp_opt_hmcts";

// static const std::string HP_OPT_604_UCT_EXPR_ID =       "604_hp_opt_uct";
// static const std::string HP_OPT_614_MAX_UCT_EXPR_ID =   "614_hp_opt_max_uct";
// static const std::string HP_OPT_624_MENTS_EXPR_ID =     "624_hp_opt_ments";
// static const std::string HP_OPT_634_BTS_EXPR_ID =       "634_hp_opt_bts";
// static const std::string HP_OPT_644_DENTS_EXPR_ID =     "644_hp_opt_dents";
// static const std::string HP_OPT_654_RENTS_EXPR_ID =     "654_hp_opt_rents";
// static const std::string HP_OPT_664_TENTS_EXPR_ID =     "664_hp_opt_tents";
// static const std::string HP_OPT_674_HMCTS_EXPR_ID =     "674_hp_opt_hmcts";

// static const std::string HP_OPT_605_UCT_EXPR_ID =       "605_hp_opt_uct";
// static const std::string HP_OPT_615_MAX_UCT_EXPR_ID =   "615_hp_opt_max_uct";
// static const std::string HP_OPT_625_MENTS_EXPR_ID =     "625_hp_opt_ments";
// static const std::string HP_OPT_635_BTS_EXPR_ID =       "635_hp_opt_bts";
// static const std::string HP_OPT_645_DENTS_EXPR_ID =     "645_hp_opt_dents";
// static const std::string HP_OPT_655_RENTS_EXPR_ID =     "655_hp_opt_rents";
// static const std::string HP_OPT_665_TENTS_EXPR_ID =     "665_hp_opt_tents";
// static const std::string HP_OPT_675_HMCTS_EXPR_ID =     "675_hp_opt_hmcts";

// static const std::string EVAL_FL_D_8x8_EXPR_ID = "800_eval_fl_d_8x8";
// static const std::string EVAL_FL_D_8x16_EXPR_ID = "801_eval_fl_d_8x16";
// static const std::string EVAL_FL_D_16x16_EXPR_ID = "802_eval_fl_d_16x16";

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
// xpr id -> env id
// ---------------------------------------------------------------------------------------------------------------------

// BOOKMARK: xpr id -> env id


// // env id lookup - helper dict to lookup env ids from hp opt experiment ids
// static const std::unordered_map<std::string,std::string> HP_OPT_EXPR_ID_TO_ENV_ID =
// {
//     {HP_OPT_600_UCT_EXPR_ID,                    FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_610_MAX_UCT_EXPR_ID,                FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_620_MENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_630_BTS_EXPR_ID,                    FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_640_DENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_650_RENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_660_TENTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},
//     {HP_OPT_670_HMCTS_EXPR_ID,                  FROZEN_LAKE_D_8x8_ENV_ID},

//     {HP_OPT_601_UCT_EXPR_ID,                    FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_611_MAX_UCT_EXPR_ID,                FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_621_MENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_631_BTS_EXPR_ID,                    FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_641_DENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_651_RENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_661_TENTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},
//     {HP_OPT_671_HMCTS_EXPR_ID,                  FROZEN_LAKE_S_8x8_ENV_ID},

//     {HP_OPT_602_UCT_EXPR_ID,                    SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_612_MAX_UCT_EXPR_ID,                SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_622_MENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_632_BTS_EXPR_ID,                    SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_642_DENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_652_RENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_662_TENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},
//     {HP_OPT_672_HMCTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID},

//     {HP_OPT_603_UCT_EXPR_ID,                    SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_613_MAX_UCT_EXPR_ID,                SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_623_MENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_633_BTS_EXPR_ID,                    SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_643_DENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_653_RENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_663_TENTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},
//     {HP_OPT_673_HMCTS_EXPR_ID,                  SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID},

//     {HP_OPT_604_UCT_EXPR_ID,                    SAILING_ENV_NORTH_ID},
//     {HP_OPT_614_MAX_UCT_EXPR_ID,                SAILING_ENV_NORTH_ID},
//     {HP_OPT_624_MENTS_EXPR_ID,                  SAILING_ENV_NORTH_ID},
//     {HP_OPT_634_BTS_EXPR_ID,                    SAILING_ENV_NORTH_ID},
//     {HP_OPT_644_DENTS_EXPR_ID,                  SAILING_ENV_NORTH_ID},
//     {HP_OPT_654_RENTS_EXPR_ID,                  SAILING_ENV_NORTH_ID},
//     {HP_OPT_664_TENTS_EXPR_ID,                  SAILING_ENV_NORTH_ID},
//     {HP_OPT_674_HMCTS_EXPR_ID,                  SAILING_ENV_NORTH_ID},

//     {HP_OPT_605_UCT_EXPR_ID,                    SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_615_MAX_UCT_EXPR_ID,                SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_625_MENTS_EXPR_ID,                  SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_635_BTS_EXPR_ID,                    SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_645_DENTS_EXPR_ID,                  SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_655_RENTS_EXPR_ID,                  SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_665_TENTS_EXPR_ID,                  SAILING_ENV_SOUTH_EAST_ID},
//     {HP_OPT_675_HMCTS_EXPR_ID,                  SAILING_ENV_SOUTH_EAST_ID},
// };































































// ---------------------------------------------------------------------------------------------------------------------
// old hpopt config written in a function :)
// ---------------------------------------------------------------------------------------------------------------------

// BOOKMARK: old hpopt config 



//     /**
//      * Gets hyperparam optimiser from expr_id
//      */
//     shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
//         string expr_id, time_t expr_timestamp, ofstream &hp_opt_summary_fs, ofstream &hp_opt_evals_fs)
//     {
//         // Params shared across optimisations (related to envs / hp_opt, and not algs themselves)
//         string env_id = HP_OPT_EXPR_ID_TO_ENV_ID.at(expr_id);
//         bool eval_wrt_time = false;
//         double search_runtime = 10000.0;
//         int max_trial_length = ENV_ID_MAX_TRIAL_LEN.at(env_id);
//         double eval_delta = 10000.0; 
//         int rollouts_per_mc_eval = 1024;
//         int num_repeats = 10; // min repeats
//         int num_threads = 16;
//         int eval_threads = 16;
//         bool use_std_mean_eval_threshold = true;

//         // Params being tuned
//         string alg_id;
//         unordered_map<string, pair<double,double>> alg_params_min_max;

//         // Defualt Q values and std_mean_eval_thresholds (default values are for sparse rewards on frozen lake envs)
//         double min_default_q_value = 0.0;
//         // if (env_id == FROZEN_LAKE_D_8x8_ENV_ID || env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID) {
//         //     min_default_q_value = -((double) max_trial_length);
//         // } else if (env_id == SAILING_ENV_NORTH_ID || env_id == SAILING_ENV_SOUTH_EAST_ID) {
//         //     min_default_q_value = -5.0 * ((double) max_trial_length);
//         // } 

//         // std mean eval thresholds
//         double std_mean_eval_threshold = 1.0; 
//         if (env_id == FROZEN_LAKE_D_8x8_ENV_ID) {
//             std_mean_eval_threshold = 1.0;
//         } else if (env_id == FROZEN_LAKE_S_8x8_ENV_ID) {
//             std_mean_eval_threshold = 0.015;
//         } else if (env_id == SLIPPY_FROZEN_LAKE_D_4x4_ENV_ID) {
//             std_mean_eval_threshold = 0.05;
//         } else if (env_id == SLIPPY_FROZEN_LAKE_S_4x4_ENV_ID) {
//             std_mean_eval_threshold = 0.005; 
//         } else if (env_id == SAILING_ENV_NORTH_ID) {
//             std_mean_eval_threshold = 0.5;
//         } else if (env_id == SAILING_ENV_SOUTH_EAST_ID) {
//             std_mean_eval_threshold = 0.5; 
//         }

//         bool mcts_mode = (env_id == SAILING_ENV_NORTH_ID || env_id == SAILING_ENV_SOUTH_EAST_ID);

//         // UCT
//         if (expr_id == HP_OPT_600_UCT_EXPR_ID 
//             || expr_id == HP_OPT_601_UCT_EXPR_ID
//             || expr_id == HP_OPT_602_UCT_EXPR_ID
//             || expr_id == HP_OPT_603_UCT_EXPR_ID
//             || expr_id == HP_OPT_604_UCT_EXPR_ID
//             || expr_id == HP_OPT_605_UCT_EXPR_ID) 
//         {
//             alg_id = UCT_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//             };
//         }
//         // MaxUCT
//         else if (expr_id == HP_OPT_610_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_611_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_612_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_613_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_614_MAX_UCT_EXPR_ID
//             || expr_id == HP_OPT_615_MAX_UCT_EXPR_ID)
//         {
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//             };
//         }
//         // MENTS
//         else if (expr_id == HP_OPT_620_MENTS_EXPR_ID
//             || expr_id == HP_OPT_621_MENTS_EXPR_ID
//             || expr_id == HP_OPT_622_MENTS_EXPR_ID
//             || expr_id == HP_OPT_623_MENTS_EXPR_ID
//             || expr_id == HP_OPT_624_MENTS_EXPR_ID
//             || expr_id == HP_OPT_625_MENTS_EXPR_ID)
//         {   
//             alg_id = MENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // BTS
//         else if (expr_id == HP_OPT_630_BTS_EXPR_ID
//             || expr_id == HP_OPT_631_BTS_EXPR_ID
//             || expr_id == HP_OPT_632_BTS_EXPR_ID
//             || expr_id == HP_OPT_633_BTS_EXPR_ID
//             || expr_id == HP_OPT_634_BTS_EXPR_ID
//             || expr_id == HP_OPT_635_BTS_EXPR_ID)
//         {
//             alg_id = BTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // DENTS
//         else if (expr_id == HP_OPT_640_DENTS_EXPR_ID
//             || expr_id == HP_OPT_641_DENTS_EXPR_ID
//             || expr_id == HP_OPT_642_DENTS_EXPR_ID
//             || expr_id == HP_OPT_643_DENTS_EXPR_ID
//             || expr_id == HP_OPT_644_DENTS_EXPR_ID
//             || expr_id == HP_OPT_645_DENTS_EXPR_ID)
//         {
//             alg_id = DENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {ENTROPY_COEFF_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {ENTROPY_DECAY_FN_PARAM_ID, make_pair(0.0, 3.0)},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, make_pair(0.01, 100.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // RENTS
//         else if (expr_id == HP_OPT_650_RENTS_EXPR_ID
//             || expr_id == HP_OPT_651_RENTS_EXPR_ID
//             || expr_id == HP_OPT_652_RENTS_EXPR_ID
//             || expr_id == HP_OPT_653_RENTS_EXPR_ID
//             || expr_id == HP_OPT_654_RENTS_EXPR_ID
//             || expr_id == HP_OPT_655_RENTS_EXPR_ID)
//         {
//             alg_id = RENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // TENTS
//         else if (expr_id == HP_OPT_660_TENTS_EXPR_ID
//             || expr_id == HP_OPT_661_TENTS_EXPR_ID
//             || expr_id == HP_OPT_662_TENTS_EXPR_ID
//             || expr_id == HP_OPT_663_TENTS_EXPR_ID
//             || expr_id == HP_OPT_664_TENTS_EXPR_ID
//             || expr_id == HP_OPT_665_TENTS_EXPR_ID)
//         {
//             alg_id = TENTS_ALG_ID;
//             alg_params_min_max = {
//                 {NORMALISE_Q_VALUES_PARAM_ID, make_pair(0.0, 1.0)},
//                 {TEMP_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {EPSILON_PARAM_ID, make_pair(0.000001, 1.0)},
//                 // {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,0.0)}
//                 {DEFAULT_Q_VALUE_PARAM_ID, make_pair(min_default_q_value,min_default_q_value)}
//             };
//         }
//         // HMCTS
//         else if (expr_id == HP_OPT_670_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_671_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_672_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_673_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_674_HMCTS_EXPR_ID
//             || expr_id == HP_OPT_675_HMCTS_EXPR_ID)
//         {
//             alg_id = HMCTS_ALG_ID;
//             alg_params_min_max = {
//                 {ADAPTIVE_BIAS_PARAM_ID, make_pair(0.0, 1.0)},
//                 {BIAS_PARAM_ID, make_pair(0.001, 1000.0)},
//                 {UCT_BUDGET_PARAM_ID, make_pair(1.0, 5000.0)},
//             };
//         }
//         // Default, haven't set up hp opt experiments for this env
//         else 
//         { 
//             stringstream ss;
//             ss << "Error in get_hyperparam_optimiser_from_expr_id for expr_id = " << expr_id;
//             throw runtime_error(ss.str());
//         }

//         return make_shared<HyperparamOptimiser>(
//             env_id,
//             expr_id,
//             expr_timestamp,
//             alg_id,
//             alg_params_min_max,
//             eval_wrt_time,
//             search_runtime,
//             max_trial_length,
//             eval_delta,
//             rollouts_per_mc_eval,
//             num_repeats,
//             num_threads,
//             eval_threads,
//             mcts_mode,
//             bo_params, 
//             hp_opt_summary_fs,
//             hp_opt_evals_fs,
//             use_std_mean_eval_threshold,
//             std_mean_eval_threshold
//         );
//     };



































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
//         // expr_id: 80x - frozen lake, deterministic, dense reward
//         // ----
//         if (expr_id == EVAL_FL_D_8x8_EXPR_ID || expr_id == EVAL_FL_D_8x16_EXPR_ID || expr_id == EVAL_FL_D_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = FROZEN_LAKE_D_8x8_ENV_ID;
//             if (expr_id == EVAL_FL_D_8x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_D_8x16_ENV_ID;
//             } else if (expr_id == EVAL_FL_D_16x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_D_16x16_ENV_ID;
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
//             // -20.8
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.01},
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
//             // -20.25
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.92},
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
//             // -21.3    
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.0085},
//                 {UCT_BUDGET_PARAM_ID, 4999},
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
//             // -18.5
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0038},
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

//             // BTS
//             // -19.36
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 72.9},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 100.0},
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

//             // DENTS
//             // -17.2
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 98.6}, // 
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 100.0},
//                 {ENTROPY_COEFF_PARAM_ID, 0.032}, //
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 0.01},
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

//             // RENTS
//             // -18.1
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.24},
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
//             // -18.8
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.077},
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

//             return run_ids;
//         }




//         // ----
//         // expr_id: 81x - frozen lake, deterministic, sparse reward
//         // ----
//         if (expr_id == EVAL_FL_S_8x8_EXPR_ID || expr_id == EVAL_FL_S_8x16_EXPR_ID || expr_id == EVAL_FL_S_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = FROZEN_LAKE_S_8x8_ENV_ID;
//             if (expr_id == EVAL_FL_S_8x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_S_8x16_ENV_ID;
//             } else if (expr_id == EVAL_FL_S_16x16_EXPR_ID) {
//                 env_id = FROZEN_LAKE_S_16x16_ENV_ID;
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
//             // 0.820304
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 2.44},
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
//             // 0.808782
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.36},
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
//             // 0.809554
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 2.75426},
//                 {UCT_BUDGET_PARAM_ID, 4995},
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
//             // 0.833115
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0022},
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
            
//             // BTS
//             // 0.833384
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.0016},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.53},
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

//             // DENTS
//             // 0.83561
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 0.0014},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.013},
//                 {ENTROPY_COEFF_PARAM_ID, 0.37},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_SQRT},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 11.9},
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

//             // RENTS
//             // 0.838962
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0011},
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
//             // 0.831394
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 0.0084},
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




//         // ----
//         // expr_id: 84x - sailing, north
//         // ----
//         if (expr_id == EVAL_SAIL_N_8x8_EXPR_ID || expr_id == EVAL_SAIL_N_8x16_EXPR_ID || expr_id == EVAL_SAIL_N_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = SAILING_ENV_NORTH_ID;
//             if (expr_id == EVAL_SAIL_N_8x16_EXPR_ID) {
//                 env_id = SAILING_8x16_ENV_NORTH_ID;
//             } else if (expr_id == EVAL_SAIL_N_16x16_EXPR_ID) {
//                 env_id = SAILING_16x16_ENV_NORTH_ID;
//             }
//             double default_q_value = -5.0 * (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
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
//             // -78.484
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 20.0638},
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
//             // -80.0736
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.599484},
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
//             // -154 (but failed)
//             // TODO
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 0.599484},
//                 {UCT_BUDGET_PARAM_ID, 1},
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
//             // -187.941
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 7.0341},
//                 {EPSILON_PARAM_ID, 0.000876699},
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
//             // -181.67
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 21.6892},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0181898},
//                 {EPSILON_PARAM_ID, 0.956878},
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
//             // -184.881
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 4.40294},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.0100076},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0142321},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 99.9819},
//                 {EPSILON_PARAM_ID, 0.999999},
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
//             // -36.1584
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 20.0744},
//                 {EPSILON_PARAM_ID, 0.998703},
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
//             // -186.15
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 25.4172},
//                 {EPSILON_PARAM_ID, 0.0590783},
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
//         // expr_id: 84x - sailing, south east
//         // ----
//         if (expr_id == EVAL_SAIL_SE_8x8_EXPR_ID || expr_id == EVAL_SAIL_SE_8x16_EXPR_ID || expr_id == EVAL_SAIL_SE_16x16_EXPR_ID)
//         {
//             // Env params
//             string env_id = SAILING_ENV_SOUTH_EAST_ID;
//             if (expr_id == EVAL_SAIL_SE_8x16_EXPR_ID) {
//                 env_id = SAILING_8x16_ENV_SOUTH_EAST_ID;
//             } else if (expr_id == EVAL_SAIL_SE_16x16_EXPR_ID) {
//                 env_id = SAILING_16x16_ENV_SOUTH_EAST_ID;
//             }
//             double default_q_value = -5.0 * (double) ENV_ID_MAX_TRIAL_LEN.at(env_id);
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
//             // -98.3836
//             alg_id = UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 20.9906},
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
//             // -90.8139
//             alg_id = MAX_UCT_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 1},
//                 {BIAS_PARAM_ID, 1.0},
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
//             // -162.195
//             // TODO
//             alg_id = HMCTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {ADAPTIVE_BIAS_PARAM_ID, 0},
//                 {BIAS_PARAM_ID, 52.5813},
//                 {UCT_BUDGET_PARAM_ID, 4990},
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
//             // -192.017
//             alg_id = MENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 5.25483},
//                 {EPSILON_PARAM_ID, 0.16076},
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
//             // -194.082
//             // TODO
//             alg_id = BTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 30.0578},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {DECAY_FN_SCALE_PARAM_ID, 0.01},
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

//             // DENTS
//             // -193.007
//             // TODO
//             alg_id = DENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 6.04038},
//                 {DECAY_FN_PARAM_ID, DECAY_FN_CONST},
//                 {DECAY_FN_SCALE_PARAM_ID, 99.9746},
//                 {ENTROPY_COEFF_PARAM_ID, 0.0960247},
//                 {ENTROPY_DECAY_FN_PARAM_ID, DECAY_FN_INV_LOG},
//                 {ENTROPY_DECAY_FN_SCALE_PARAM_ID, 99.9716},
//                 {EPSILON_PARAM_ID, 0.000167246},
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
//             // -73.9193
//             alg_id = RENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 0},
//                 {TEMP_PARAM_ID, 26.8291},
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
//             // -196.061
//             alg_id = TENTS_ALG_ID;
//             alg_params = 
//             {
//                 {MCTS_MODE_PARAM_ID, 1},
//                 {NORMALISE_Q_VALUES_PARAM_ID, 1},
//                 {TEMP_PARAM_ID, 26.721},
//                 {EPSILON_PARAM_ID, 0.0001},
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




//         stringstream ss;
//         ss << "Error in get_run_ids_from_expr_id for expr_id = " << expr_id;
//         throw runtime_error(ss.str());
//     }