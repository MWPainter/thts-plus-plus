







// ---------------------------------------------------------------------------------------------------------------------
// Deterministic envs
// ---------------------------------------------------------------------------------------------------------------------

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



// // expr ids - debug
// static const std::string DEBUG_EXPR_ID = "000_debug";

// // expr ids - supp experiments (1xx + 2xx + 3xx)
// // supp experiments = showing how performance varies with parameters etc
// static const std::string SUPP_100_DCHAIN_10_TEMP_EXPR_ID = "100_supp_dchain_temp_vary";
// static const std::string SUPP_101_MOD_DCHAIN_10_TEMP_EXPR_ID = "101_supp_mod_dchain_temp_vary";
// static const std::string SUPP_102_ENTROPY_TRAP_10_TEMP_EXPR_ID = "102_supp_entropy_temp_10_vary";
// static const std::string SUPP_103_ENTROPY_TRAP_15_TEMP_EXPR_ID = "103_supp_entropy_temp_15_vary";
// // TODO: what about the exploration param - make an expr or two for this.
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