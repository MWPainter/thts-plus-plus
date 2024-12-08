#pragma once

#include "mo/mo_thts_env.h"
#include "mo/mo_thts_manager.h"
#include "mo/mo_thts_decision_node.h"

#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "bayesopt/bayesopt.hpp"
#include "bayesopt/parameters.hpp"

// env ids - debug
static const std::string DEBUG_ENV_1_ID = "debug_env_1"; // not stoch + 2 rew
static const std::string DEBUG_ENV_2_ID = "debug_env_2"; // stoch + 2 rew
static const std::string DEBUG_ENV_3_ID = "debug_env_3"; // not stoch + 4 rew
static const std::string DEBUG_ENV_4_ID = "debug_env_4"; // stoch + 4 rew

static const std::unordered_set<std::string> DEBUG_ENVS =
{
     DEBUG_ENV_1_ID,
     DEBUG_ENV_2_ID,
     DEBUG_ENV_3_ID,
     DEBUG_ENV_4_ID,
};

static const std::string DEBUG_PY_ENV_1_ID = "py_debug_env_1"; // not stoch + 2 rew
static const std::string DEBUG_PY_ENV_2_ID = "py_debug_env_2"; // stoch + 2 rew
static const std::string DEBUG_PY_ENV_3_ID = "py_debug_env_3"; // not stoch + 4 rew
static const std::string DEBUG_PY_ENV_4_ID = "py_debug_env_4"; // stoch + 4 rew

static const std::unordered_set<std::string> DEBUG_PY_ENVS =
{
     DEBUG_PY_ENV_1_ID,
     DEBUG_PY_ENV_2_ID,
     DEBUG_PY_ENV_3_ID,
     DEBUG_PY_ENV_4_ID,
};

// env ids - toy/tree
// TODO: implement + integrate tree envs to demonstrate things
static const std::unordered_set<std::string> TOY_ENVS =
{
};

// env ids - mo gymnasium (discr obs / discr act)
static const std::string DST_ENV_ID = "deep-sea-treasure-v0";                   // unused (using improved DST vamplew wrapper)            
static const std::string DST_CONC_ENV_ID = "deep-sea-treasure-concave-v0";      // unused (using improved DST vamplew wrapper)
static const std::string DST_MIRR_ENV_ID = "deep-sea-treasure-mirrored-v0";     // unused (using improved DST vamplew wrapper)
static const std::string RESOURCE_GATHER_ENV_ID = "resource-gathering-v0";      // 38x + 68x
static const std::string BREAKABLE_BOTTLES_ENV_ID = "breakable-bottles-v0";     // 37x + 67x
static const std::string FRUIT_TREE_ENV_ID = "fruit-tree-v0";                   // unused (using custom copy with opt for stoch)
static const std::string FOUR_ROOM_ENV_ID = "four-room-v0";                     // 40x + 70x

// env ids - mo gymnasium (cts obs / discr act)
static const std::string MOUNTAIN_CAR_ENV_ID = "mo-mountaincar-v0";             
static const std::string LUNAR_LANDER_ENV_ID = "mo-lunar-lander-v2";         
static const std::string MINECART_DETERMINISTIC_ENV_ID = "minecart-deterministic-v0";       // 50x + 80x      
static const std::string MINECART_ENV_ID = "minecart-v0";                                   // 51x + 81x
static const std::string HIGHWAY_ENV_ID = "mo-highway-v0";                 

// list of mo gymnasium envs
static const std::unordered_set<std::string> MO_GYM_ENVS =
{
    DST_ENV_ID,
    DST_CONC_ENV_ID,
    DST_MIRR_ENV_ID,
    RESOURCE_GATHER_ENV_ID,
    BREAKABLE_BOTTLES_ENV_ID,
    FRUIT_TREE_ENV_ID,
    FOUR_ROOM_ENV_ID,
    MOUNTAIN_CAR_ENV_ID,
    LUNAR_LANDER_ENV_ID,
    MINECART_DETERMINISTIC_ENV_ID,
    MINECART_ENV_ID,
    HIGHWAY_ENV_ID,
};           

// env ids - mo gymnasium (with extra time cost)
static const std::string RESOURCE_GATHER_TIMED_ENV_ID = "resource-gathering-timed-v0";      // 39x + 69x
static const std::string FOUR_ROOM_TIMED_ENV_ID = "four-room-timed-v0";                     // 41x + 71x

// list of envs - (timed) mo gymnasium envs (with map to underlying gym env)           
static const std::unordered_map<std::string,std::string> TIMED_ENV_ID_TO_GYM_ID =
{
    {RESOURCE_GATHER_TIMED_ENV_ID, RESOURCE_GATHER_ENV_ID},
    {FOUR_ROOM_TIMED_ENV_ID, FOUR_ROOM_ENV_ID},
};

// env ids - custom python envs (adaptations of mo gym envs)
static const std::string FRUIT_TREE_7_ENV_ID = "fruit-tree-depth-7";                    // 34x + 64x
static const std::string FRUIT_TREE_STOCH_5_ENV_ID = "fruit-tree-stoch-depth-5";        // 35x + 65x
static const std::string FRUIT_TREE_STOCH_7_ENV_ID = "fruit-tree-stoch-depth-7";        // 36x + 66x

static const std::string IMPROVED_DST_ENV_ID = "deep-sea-treasure-improved";                // 30x + 60x
static const std::string IMPROVED_STOCH_DST_ENV_ID = "deep-sea-treasure-improved-stoch";    // 31x + 61x
static const std::string VAMPLEW_DST_ENV_ID = "deep-sea-treasure-vamplew";                  // 32x + 62x
static const std::string VAMPLEW_STOCH_DST_ENV_ID = "deep-sea-treasure-vamplew-stoch";      // 33x + 63x

static const std::unordered_set<std::string> PY_ENVS =
{
    FRUIT_TREE_7_ENV_ID,
    FRUIT_TREE_STOCH_5_ENV_ID,
    FRUIT_TREE_STOCH_7_ENV_ID,
    IMPROVED_DST_ENV_ID,
    IMPROVED_STOCH_DST_ENV_ID,
    VAMPLEW_DST_ENV_ID,
    VAMPLEW_STOCH_DST_ENV_ID,
};

// env ids - max trial length
static const std::unordered_map<std::string,int> ENV_ID_MAX_TRIAL_LEN = 
{
    {DST_ENV_ID,50},
    {DST_CONC_ENV_ID,50},
    {DST_MIRR_ENV_ID,50},
    {RESOURCE_GATHER_ENV_ID,50},
    {RESOURCE_GATHER_TIMED_ENV_ID,50},
    {BREAKABLE_BOTTLES_ENV_ID,50},
    {FRUIT_TREE_ENV_ID,50},
    {FOUR_ROOM_ENV_ID,50},
    {FOUR_ROOM_TIMED_ENV_ID,50},
    {MOUNTAIN_CAR_ENV_ID,50},
    {LUNAR_LANDER_ENV_ID,50},
    {MINECART_DETERMINISTIC_ENV_ID,50},
    {MINECART_ENV_ID,50},
    {HIGHWAY_ENV_ID,50},
    {FRUIT_TREE_7_ENV_ID,50},
    {FRUIT_TREE_STOCH_5_ENV_ID,50},
    {FRUIT_TREE_STOCH_7_ENV_ID,50},
    {IMPROVED_DST_ENV_ID,50},
    {IMPROVED_STOCH_DST_ENV_ID,50},
    {VAMPLEW_DST_ENV_ID,50},
    {VAMPLEW_STOCH_DST_ENV_ID,50},
};

// alg ids 
static const std::string CZT_ALG_ID = "czt";
static const std::string CHMCTS_ALG_ID = "chmcts";
static const std::string SMBTS_ALG_ID = "smbts";
static const std::string SMDENTS_ALG_ID = "smdents";

// expr ids - testing - for VS debugging
static const std::string DEBUG_EXPR_ID = "000_debug";

// expr ids - testing - debugging 
static const std::string DEBUG_ENV_1_EXPR_ID = "001_debug_env_1";
static const std::string DEBUG_PY_ENV_1_EXPR_ID = "002_debug_py_env_1";
static const std::string DEBUG_ENV_2_EXPR_ID = "003_debug_env_2";
static const std::string DEBUG_PY_ENV_2_EXPR_ID = "004_debug_py_env_2";
static const std::string DEBUG_ENV_3_EXPR_ID = "005_debug_env_3";
static const std::string DEBUG_PY_ENV_3_EXPR_ID = "006_debug_py_env_3";
static const std::string DEBUG_ENV_4_EXPR_ID = "007_debug_env_4";
static const std::string DEBUG_PY_ENV_4_EXPR_ID = "008_debug_py_env_4";

// expr ids - testing - proof of concept tests on mo-gym envs (running without hyperparam tuning)
static const std::string POC_DST_EXPR_ID = "009_poc_dst";
static const std::string POC_FT_EXPR_ID = "010_poc_ft";

// expr ids - testing - debugging HP opt
static const std::string DEBUG_CZT_HP_OPT_EXPR_ID = "020_debug_czt_hp";
static const std::string DEBUG_CHMCTS_HP_OPT_EXPR_ID = "021_debug_chmcts_hp";
static const std::string DEBUG_SMBTS_HP_OPT_EXPR_ID = "022_debug_smbts_hp";
static const std::string DEBUG_SMDENTS_HP_OPT_EXPR_ID = "023_debug_smdents_hp";

// expr ids - toy/tree (1xx = hyperparam tuning)
// TODO


// expr ids - toy/tree (2xx = eval)
// TODO

// expr ids - mo gymnasium + custom envs (3xx + 4xx + 5xx = hyperparam tuning)
// - deep sea treasure
static const std::string HP_OPT_DST_CZT_EXPR_ID = "300_hp_opt_deep_sea_treasure_czt";
static const std::string HP_OPT_DST_CHMCTS_EXPR_ID = "301_hp_opt_deep_sea_treasure_chmcts";
static const std::string HP_OPT_DST_SMBTS_EXPR_ID = "302_hp_opt_deep_sea_treasure_smbts";
static const std::string HP_OPT_DST_SMDENTS_EXPR_ID = "303_hp_opt_deep_sea_treasure_smdents";
// - deep sea treasure (stochastic)
static const std::string HP_OPT_DST_STOCH_CZT_EXPR_ID = "310_hp_opt_deep_sea_treasure_stoch_czt";
static const std::string HP_OPT_DST_STOCH_CHMCTS_EXPR_ID = "311_hp_opt_deep_sea_treasure_stoch_chmcts";
static const std::string HP_OPT_DST_STOCH_SMBTS_EXPR_ID = "312_hp_opt_deep_sea_treasure_stoch_smbts";
static const std::string HP_OPT_DST_STOCH_SMDENTS_EXPR_ID = "313_hp_opt_deep_sea_treasure_stoch_smdents";
// - improved deep sea treasure
static const std::string HP_OPT_DST_IMPR_CZT_EXPR_ID = "320_hp_opt_deep_sea_treasure_impr_czt";
static const std::string HP_OPT_DST_IMPR_CHMCTS_EXPR_ID = "321_hp_opt_deep_sea_treasure_impr_chmcts";
static const std::string HP_OPT_DST_IMPR_SMBTS_EXPR_ID = "322_hp_opt_deep_sea_treasure_impr_smbts";
static const std::string HP_OPT_DST_IMPR_SMDENTS_EXPR_ID = "323_hp_opt_deep_sea_treasure_impr_smdents";
// - improved deep sea treasure (stochastic)
static const std::string HP_OPT_DST_IMPR_STOCH_CZT_EXPR_ID = "330_hp_opt_deep_sea_treasure_impr_stoch_czt";
static const std::string HP_OPT_DST_IMPR_STOCH_CHMCTS_EXPR_ID = "331_hp_opt_deep_sea_treasure_impr_stoch_chmcts";
static const std::string HP_OPT_DST_IMPR_STOCH_SMBTS_EXPR_ID = "332_hp_opt_deep_sea_treasure_impr_stoch_smbts";
static const std::string HP_OPT_DST_IMPR_STOCH_SMDENTS_EXPR_ID = "333_hp_opt_deep_sea_treasure_impr_stoch_smdents";
// - fruit tree
static const std::string HP_OPT_FT_CZT_EXPR_ID = "340_hp_opt_fruit_tree_czt";
static const std::string HP_OPT_FT_CHMCTS_EXPR_ID = "341_hp_opt_fruit_tree_chmcts";
static const std::string HP_OPT_FT_SMBTS_EXPR_ID = "342_hp_opt_fruit_tree_smbts";
static const std::string HP_OPT_FT_SMDENTS_EXPR_ID = "343_hp_opt_fruit_tree_smdents";
// - fruit tree (stochastic, depth 5)
static const std::string HP_OPT_FT_S5_CZT_EXPR_ID = "350_hp_opt_fruit_tree_stoch_5_czt";
static const std::string HP_OPT_FT_S5_CHMCTS_EXPR_ID = "351_hp_opt_fruit_tree_stoch_5_chmcts";
static const std::string HP_OPT_FT_S5_SMBTS_EXPR_ID = "352_hp_opt_fruit_tree_stoch_5_smbts";
static const std::string HP_OPT_FT_S5_SMDENTS_EXPR_ID = "353_hp_opt_fruit_tree_stoch_5_smdents";
// - fruit tree (stochastic, depth 7)
static const std::string HP_OPT_FT_S7_CZT_EXPR_ID = "360_hp_opt_fruit_tree_stoch_7_czt";
static const std::string HP_OPT_FT_S7_CHMCTS_EXPR_ID = "361_hp_opt_fruit_tree_stoch_7_chmcts";
static const std::string HP_OPT_FT_S7_SMBTS_EXPR_ID = "362_hp_opt_fruit_tree_stoch_7_smbts";
static const std::string HP_OPT_FT_S7_SMDENTS_EXPR_ID = "363_hp_opt_fruit_tree_stoch_7_smdents";
// - breakable bottles
static const std::string HP_OPT_BB_CZT_EXPR_ID = "370_hp_opt_breakable_bottles_czt";
static const std::string HP_OPT_BB_CHMCTS_EXPR_ID = "371_hp_opt_breakable_bottles_chmcts";
static const std::string HP_OPT_BB_SMBTS_EXPR_ID = "372_hp_opt_breakable_bottles_smbts";
static const std::string HP_OPT_BB_SMDENTS_EXPR_ID = "373_hp_opt_breakable_bottles_smdents";
// - resource gathering
static const std::string HP_OPT_RG_CZT_EXPR_ID = "380_hp_opt_resource_gathering_czt";
static const std::string HP_OPT_RG_CHMCTS_EXPR_ID = "381_hp_opt_resource_gathering_chmcts";
static const std::string HP_OPT_RG_SMBTS_EXPR_ID = "382_hp_opt_resource_gathering_smbts";
static const std::string HP_OPT_RG_SMDENTS_EXPR_ID = "383_hp_opt_resource_gathering_smdents";
// - resource gathering - timed
static const std::string HP_OPT_RG_TIMED_CZT_EXPR_ID = "390_hp_opt_resource_gathering_timed_czt";
static const std::string HP_OPT_RG_TIMED_CHMCTS_EXPR_ID = "391_hp_opt_resource_gathering_timed_chmcts";
static const std::string HP_OPT_RG_TIMED_SMBTS_EXPR_ID = "392_hp_opt_resource_gathering_timed_smbts";
static const std::string HP_OPT_RG_TIMED_SMDENTS_EXPR_ID = "393_hp_opt_resource_gathering_timed_smdents";
// - four room
static const std::string HP_OPT_4R_CZT_EXPR_ID = "400_hp_opt_four_room_czt";
static const std::string HP_OPT_4R_CHMCTS_EXPR_ID = "401_hp_opt_four_room_chmcts";
static const std::string HP_OPT_4R_SMBTS_EXPR_ID = "402_hp_opt_four_room_smbts";
static const std::string HP_OPT_4R_SMDENTS_EXPR_ID = "403_hp_opt_four_room_smdents";
// - four room - timed
static const std::string HP_OPT_4R_TIMED_CZT_EXPR_ID = "410_hp_opt_four_room_timed_czt";
static const std::string HP_OPT_4R_TIMED_CHMCTS_EXPR_ID = "411_hp_opt_four_room_timed_chmcts";
static const std::string HP_OPT_4R_TIMED_SMBTS_EXPR_ID = "412_hp_opt_four_room_timed_smbts";
static const std::string HP_OPT_4R_TIMED_SMDENTS_EXPR_ID = "413_hp_opt_four_room_timed_smdents";
// - minecart - deterministic
static const std::string HP_OPT_MINECART_DET_CZT_EXPR_ID = "500_hp_opt_minecart_deterministic_czt";
static const std::string HP_OPT_MINECART_DET_CHMCTS_EXPR_ID = "501_hp_opt_minecart_deterministic_chmcts";
static const std::string HP_OPT_MINECART_DET_SMBTS_EXPR_ID = "502_hp_opt_minecart_deterministic_smbts";
static const std::string HP_OPT_MINECART_DET_SMDENTS_EXPR_ID = "503_hp_opt_minecart_deterministic_smdents";
// - minecart 
static const std::string HP_OPT_MINECART_CZT_EXPR_ID = "510_hp_opt_minecart_czt";
static const std::string HP_OPT_MINECART_CHMCTS_EXPR_ID = "511_hp_opt_minecart_chmcts";
static const std::string HP_OPT_MINECART_SMBTS_EXPR_ID = "512_hp_opt_minecart_smbts";
static const std::string HP_OPT_MINECART_SMDENTS_EXPR_ID = "513_hp_opt_minecart_smdents";

// expr ids - mo gymnasium + custom envs (6xx + 7xx + 8xx = eval)
static const std::string EVAL_DST_EXPR_ID = "600_dst";

// expr ids - lists of czt / chmcts / bts / dents expr_ids
static const std::unordered_map<std::string,std::string> HP_OPT_MOGYM_CZT_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_CZT_EXPR_ID,                VAMPLEW_DST_ENV_ID},
    {HP_OPT_DST_STOCH_CZT_EXPR_ID,          VAMPLEW_STOCH_DST_ENV_ID},
    {HP_OPT_DST_IMPR_CZT_EXPR_ID,           IMPROVED_DST_ENV_ID},
    {HP_OPT_DST_IMPR_STOCH_CZT_EXPR_ID,     IMPROVED_STOCH_DST_ENV_ID},
    {HP_OPT_FT_CZT_EXPR_ID,                 FRUIT_TREE_7_ENV_ID},
    {HP_OPT_FT_S5_CZT_EXPR_ID,              FRUIT_TREE_STOCH_5_ENV_ID},
    {HP_OPT_FT_S7_CZT_EXPR_ID,              FRUIT_TREE_STOCH_7_ENV_ID},
    {HP_OPT_BB_CZT_EXPR_ID,                 BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_RG_CZT_EXPR_ID,                 RESOURCE_GATHER_ENV_ID},
    {HP_OPT_RG_TIMED_CZT_EXPR_ID,           RESOURCE_GATHER_TIMED_ENV_ID},
    {HP_OPT_4R_CZT_EXPR_ID,                 FOUR_ROOM_ENV_ID},
    {HP_OPT_4R_TIMED_CZT_EXPR_ID,           FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINECART_DET_CZT_EXPR_ID,       MINECART_DETERMINISTIC_ENV_ID},
    {HP_OPT_MINECART_CZT_EXPR_ID,           MINECART_ENV_ID},
};
static const std::unordered_map<std::string,std::string> HP_OPT_MOGYM_CHMCTS_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_CHMCTS_EXPR_ID,                VAMPLEW_DST_ENV_ID},
    {HP_OPT_DST_STOCH_CHMCTS_EXPR_ID,          VAMPLEW_STOCH_DST_ENV_ID},
    {HP_OPT_DST_IMPR_CHMCTS_EXPR_ID,           IMPROVED_DST_ENV_ID},
    {HP_OPT_DST_IMPR_STOCH_CHMCTS_EXPR_ID,     IMPROVED_STOCH_DST_ENV_ID},
    {HP_OPT_FT_CHMCTS_EXPR_ID,                 FRUIT_TREE_7_ENV_ID},
    {HP_OPT_FT_S5_CHMCTS_EXPR_ID,              FRUIT_TREE_STOCH_5_ENV_ID},
    {HP_OPT_FT_S7_CHMCTS_EXPR_ID,              FRUIT_TREE_STOCH_7_ENV_ID},
    {HP_OPT_BB_CHMCTS_EXPR_ID,                 BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_RG_CHMCTS_EXPR_ID,                 RESOURCE_GATHER_ENV_ID},
    {HP_OPT_RG_TIMED_CHMCTS_EXPR_ID,           RESOURCE_GATHER_TIMED_ENV_ID},
    {HP_OPT_4R_CHMCTS_EXPR_ID,                 FOUR_ROOM_ENV_ID},
    {HP_OPT_4R_TIMED_CHMCTS_EXPR_ID,           FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINECART_DET_CHMCTS_EXPR_ID,       MINECART_DETERMINISTIC_ENV_ID},
    {HP_OPT_MINECART_CHMCTS_EXPR_ID,           MINECART_ENV_ID},
};
static const std::unordered_map<std::string,std::string> HP_OPT_MOGYM_SMBTS_EXPR_ID_TO_ENV_ID =
{   
    {HP_OPT_DST_SMBTS_EXPR_ID,                VAMPLEW_DST_ENV_ID},
    {HP_OPT_DST_STOCH_SMBTS_EXPR_ID,          VAMPLEW_STOCH_DST_ENV_ID},
    {HP_OPT_DST_IMPR_SMBTS_EXPR_ID,           IMPROVED_DST_ENV_ID},
    {HP_OPT_DST_IMPR_STOCH_SMBTS_EXPR_ID,     IMPROVED_STOCH_DST_ENV_ID},
    {HP_OPT_FT_SMBTS_EXPR_ID,                 FRUIT_TREE_7_ENV_ID},
    {HP_OPT_FT_S5_SMBTS_EXPR_ID,              FRUIT_TREE_STOCH_5_ENV_ID},
    {HP_OPT_FT_S7_SMBTS_EXPR_ID,              FRUIT_TREE_STOCH_7_ENV_ID},
    {HP_OPT_BB_SMBTS_EXPR_ID,                 BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_RG_SMBTS_EXPR_ID,                 RESOURCE_GATHER_ENV_ID},
    {HP_OPT_RG_TIMED_SMBTS_EXPR_ID,           RESOURCE_GATHER_TIMED_ENV_ID},
    {HP_OPT_4R_SMBTS_EXPR_ID,                 FOUR_ROOM_ENV_ID},
    {HP_OPT_4R_TIMED_SMBTS_EXPR_ID,           FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINECART_DET_SMBTS_EXPR_ID,       MINECART_DETERMINISTIC_ENV_ID},
    {HP_OPT_MINECART_SMBTS_EXPR_ID,           MINECART_ENV_ID},
};
static const std::unordered_map<std::string,std::string> HP_OPT_MOGYM_SMDENTS_EXPR_ID_TO_ENV_ID =
{
    {HP_OPT_DST_SMDENTS_EXPR_ID,                VAMPLEW_DST_ENV_ID},
    {HP_OPT_DST_STOCH_SMDENTS_EXPR_ID,          VAMPLEW_STOCH_DST_ENV_ID},
    {HP_OPT_DST_IMPR_SMDENTS_EXPR_ID,           IMPROVED_DST_ENV_ID},
    {HP_OPT_DST_IMPR_STOCH_SMDENTS_EXPR_ID,     IMPROVED_STOCH_DST_ENV_ID},
    {HP_OPT_FT_SMDENTS_EXPR_ID,                 FRUIT_TREE_7_ENV_ID},
    {HP_OPT_FT_S5_SMDENTS_EXPR_ID,              FRUIT_TREE_STOCH_5_ENV_ID},
    {HP_OPT_FT_S7_SMDENTS_EXPR_ID,              FRUIT_TREE_STOCH_7_ENV_ID},
    {HP_OPT_BB_SMDENTS_EXPR_ID,                 BREAKABLE_BOTTLES_ENV_ID},
    {HP_OPT_RG_SMDENTS_EXPR_ID,                 RESOURCE_GATHER_ENV_ID},
    {HP_OPT_RG_TIMED_SMDENTS_EXPR_ID,           RESOURCE_GATHER_TIMED_ENV_ID},
    {HP_OPT_4R_SMDENTS_EXPR_ID,                 FOUR_ROOM_ENV_ID},
    {HP_OPT_4R_TIMED_SMDENTS_EXPR_ID,           FOUR_ROOM_TIMED_ENV_ID},
    {HP_OPT_MINECART_DET_SMDENTS_EXPR_ID,       MINECART_DETERMINISTIC_ENV_ID},
    {HP_OPT_MINECART_SMDENTS_EXPR_ID,           MINECART_ENV_ID},
};

// list of all expr ids
static const std::unordered_set<std::string> ALL_EXPR_IDS = 
{
    DEBUG_EXPR_ID,
    DEBUG_ENV_1_EXPR_ID,
    DEBUG_PY_ENV_1_EXPR_ID,
    DEBUG_ENV_2_EXPR_ID,
    DEBUG_PY_ENV_2_EXPR_ID,
    DEBUG_ENV_3_EXPR_ID,
    DEBUG_PY_ENV_3_EXPR_ID,
    DEBUG_ENV_4_EXPR_ID,
    DEBUG_PY_ENV_4_EXPR_ID,
    POC_DST_EXPR_ID,
    POC_FT_EXPR_ID,
    DEBUG_CZT_HP_OPT_EXPR_ID,
    DEBUG_CHMCTS_HP_OPT_EXPR_ID,
    DEBUG_SMBTS_HP_OPT_EXPR_ID,
    DEBUG_SMDENTS_HP_OPT_EXPR_ID,

    HP_OPT_DST_CZT_EXPR_ID,
    HP_OPT_DST_CHMCTS_EXPR_ID,
    HP_OPT_DST_SMBTS_EXPR_ID,
    HP_OPT_DST_SMDENTS_EXPR_ID,
    HP_OPT_DST_STOCH_CZT_EXPR_ID,
    HP_OPT_DST_STOCH_CHMCTS_EXPR_ID,
    HP_OPT_DST_STOCH_SMBTS_EXPR_ID,
    HP_OPT_DST_STOCH_SMDENTS_EXPR_ID,
    HP_OPT_DST_IMPR_CZT_EXPR_ID,
    HP_OPT_DST_IMPR_CHMCTS_EXPR_ID,
    HP_OPT_DST_IMPR_SMBTS_EXPR_ID,
    HP_OPT_DST_IMPR_SMDENTS_EXPR_ID,
    HP_OPT_DST_IMPR_STOCH_CZT_EXPR_ID,
    HP_OPT_DST_IMPR_STOCH_CHMCTS_EXPR_ID,
    HP_OPT_DST_IMPR_STOCH_SMBTS_EXPR_ID,
    HP_OPT_DST_IMPR_STOCH_SMDENTS_EXPR_ID,
    HP_OPT_FT_CZT_EXPR_ID,
    HP_OPT_FT_CHMCTS_EXPR_ID,
    HP_OPT_FT_SMBTS_EXPR_ID,
    HP_OPT_FT_SMDENTS_EXPR_ID,
    HP_OPT_FT_S5_CZT_EXPR_ID,
    HP_OPT_FT_S5_CHMCTS_EXPR_ID,
    HP_OPT_FT_S5_SMBTS_EXPR_ID,
    HP_OPT_FT_S5_SMDENTS_EXPR_ID,
    HP_OPT_FT_S7_CZT_EXPR_ID,
    HP_OPT_FT_S7_CHMCTS_EXPR_ID,
    HP_OPT_FT_S7_SMBTS_EXPR_ID,
    HP_OPT_FT_S7_SMDENTS_EXPR_ID,
    HP_OPT_BB_CZT_EXPR_ID,
    HP_OPT_BB_CHMCTS_EXPR_ID,
    HP_OPT_BB_SMBTS_EXPR_ID,
    HP_OPT_BB_SMDENTS_EXPR_ID,
    HP_OPT_RG_CZT_EXPR_ID,
    HP_OPT_RG_CHMCTS_EXPR_ID,
    HP_OPT_RG_SMBTS_EXPR_ID,
    HP_OPT_RG_SMDENTS_EXPR_ID,
    HP_OPT_RG_TIMED_CZT_EXPR_ID,
    HP_OPT_RG_TIMED_CHMCTS_EXPR_ID,
    HP_OPT_RG_TIMED_SMBTS_EXPR_ID,
    HP_OPT_RG_TIMED_SMDENTS_EXPR_ID,
    HP_OPT_4R_CZT_EXPR_ID,
    HP_OPT_4R_CHMCTS_EXPR_ID,
    HP_OPT_4R_SMBTS_EXPR_ID,
    HP_OPT_4R_SMDENTS_EXPR_ID,
    HP_OPT_4R_TIMED_CZT_EXPR_ID,
    HP_OPT_4R_TIMED_CHMCTS_EXPR_ID,
    HP_OPT_4R_TIMED_SMBTS_EXPR_ID,
    HP_OPT_4R_TIMED_SMDENTS_EXPR_ID,
    HP_OPT_MINECART_DET_CZT_EXPR_ID,
    HP_OPT_MINECART_DET_CHMCTS_EXPR_ID,
    HP_OPT_MINECART_DET_SMBTS_EXPR_ID,
    HP_OPT_MINECART_DET_SMDENTS_EXPR_ID,
    HP_OPT_MINECART_CZT_EXPR_ID,
    HP_OPT_MINECART_CHMCTS_EXPR_ID,
    HP_OPT_MINECART_SMBTS_EXPR_ID,
    HP_OPT_MINECART_SMDENTS_EXPR_ID,

    EVAL_DST_EXPR_ID,
};

// param ids
static const std::string CZT_BIAS_PARAM_ID = "czt_bias";
static const std::string CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID = "czt_ball_split_visit_thresh";

static const std::string SM_L_INF_THRESH_PARAM_ID = "sm_l_inf_thresh";
static const std::string SM_MAX_DEPTH = "sm_max_depth";
static const std::string SM_SPLIT_VISIT_THRESH_PARAM_ID = "sm_split_visit_thresh";

static const std::string SMBTS_SEARCH_TEMP_PARAM_ID = "smbts_search_temp";
static const std::string SMBTS_EPSILON_PARAM_ID = "smbts_epsilon";
static const std::string SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID = "smbts_use_search_temp_decay";
static const std::string SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID = "smbts_search_temp_decay_visits_scale";

static const std::string SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID = "smdents_entropy_temp_init";
static const std::string SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID = "smdents_entropy_temp_visits_scale";

// relevant alg ids -> param ids
static const std::unordered_map<std::string,std::vector<std::string>> RELEVANT_PARAM_IDS =
{
    {CZT_ALG_ID,
        {
            CZT_BIAS_PARAM_ID,
            CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
        },
    },
    {CHMCTS_ALG_ID,
        {
            CZT_BIAS_PARAM_ID,
            CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
        },
    },
    {SMBTS_ALG_ID,
        {
            SM_L_INF_THRESH_PARAM_ID,
            // SM_MAX_DEPTH,
            SM_SPLIT_VISIT_THRESH_PARAM_ID,
            SMBTS_SEARCH_TEMP_PARAM_ID,
            SMBTS_EPSILON_PARAM_ID,
            SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
            SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID,
        },
    },
    {SMDENTS_ALG_ID,
        {
            SM_L_INF_THRESH_PARAM_ID,
            // SM_MAX_DEPTH,
            SM_SPLIT_VISIT_THRESH_PARAM_ID,
            SMBTS_SEARCH_TEMP_PARAM_ID,
            SMBTS_EPSILON_PARAM_ID,
            SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
            SMBTS_SEARCH_TEMP_DECAY_VISITS_SCALE_PARAM_ID,
            SMDENTS_ENTROPY_TEMP_INIT_PARAM_ID,
            SMDENTS_ENTROPY_TEMP_VISITS_SCALE_PARAM_ID
        },
    },
};

// List of boolean + int param ids
static const std::unordered_set<std::string> BOOLEAN_PARAM_IDS =
{
    SMBTS_SEARCH_TEMP_USE_DECAY_PARAM_ID,
};

static const std::unordered_set<std::string> INTEGER_PARAM_IDS =
{
    CZT_BALL_SPLIT_VISIT_THRESH_PARAM_ID,
    // SM_MAX_DEPTH,
    SM_SPLIT_VISIT_THRESH_PARAM_ID,
};


namespace thts {
    /**
     * Lookup expr_id from prefix
     */
    std::string lookup_expr_id_from_prefix(std::string expr_id_prefix);

    /**
     * Create the env corresponding to 'env_id' and return is
     */
    std::shared_ptr<MoThtsEnv> get_env(std::string env_id);

    /**
     * Checks if env corresponding to 'env_id' is a python env
     */
    bool is_python_env(std::string env_id);

    /**
     * The minimum value possible in the environment
     * Useful for setting default values
     * (Note min value for each independent objective, which isn't necessarily achievable)
    */
    Eigen::ArrayXd get_env_min_value(std::string env_id, int max_trial_length);

    /**
     * Returns the max value possible in the environment 
     * (Same as get_env_min_value, but for maximum)
    */
    Eigen::ArrayXd get_env_max_value(std::string env_id, int max_trial_length);

    /**
     * Struct to wrap all the params for a eval run
     * 
     * Member variables - env/alg params:
     *      env_id: A string id for an environment instance
     *      expr_id: An id for the current experiment being run
     *      expr_timestamp: A timestamp to mark expr_id with (so can rerun same expr without overwriting results)
     *      alg_params: dictionary of alg params below
     *      czt_bias: czt / chmcts param
     *      czt_ball_split_visit_thresh: czt / chmcts param
     *      sm_l_inf_thresh: simplex map param
     *      sm_max_depth: simplex map param
     *      sm_split_visit_thresh: simplex map param
     *      smbts_search_temp: smbts param
     *      smbts_epsilon: smbts param
     *      smbts_use_search_temp_decay: smbts param
     *      smbts_search_temp_decay_visits_scale: smbts param
     *      smdents_entropy_temp_init: smdents param
     *      smdents_entropy_temp_visits_scale: smdents param
     * 
     * Member variables - tree search params:
     *      search_runtime: The total runtime to use for each search (in seconds)
     *      max_trial_length: The maximum trial length to use for the run
     *      eval_delta: The frequency of logging/running mc eval to use (in seconds)
     *      rollouts_per_mc_eval: How many trials to use for mc evals   
     *      num_repeats: The number of times that this run should be repeated
     *      num_threads: The number of threads to use tree search
     *      eval_threads: The number of threads to use in mc evals
     *      num_envs: The number of environments for ThtsManager to duplicate
    */
    struct RunID {
        public:
            std::string env_id;
            std::string expr_id;
            std::time_t expr_timestamp;
            std::string alg_id;

            std::unordered_map<std::string, double> alg_params;

            double czt_bias;
            int czt_ball_split_visit_thresh;
            double sm_l_inf_thresh;
            int sm_max_depth;
            int sm_split_visit_thresh;
            double smbts_search_temp;
            double smbts_epsilon;
            bool smbts_use_search_temp_decay;
            double smbts_search_temp_decay_visits_scale;
            double smdents_entropy_temp_init;
            double smdents_entropy_temp_visits_scale;

            double search_runtime;
            int max_trial_length;
            double eval_delta;
            int rollouts_per_mc_eval;
            int num_repeats;
            int num_threads;
            int eval_threads;
            int num_envs;

            /**
             * Default constructor
            */
            RunID();

            /**
             * Initialised constructor
            */
            RunID(
                std::string env_id,
                std::string expr_id,
                std::time_t expr_timestamp,
                std::string alg_id,
                std::unordered_map<std::string, double>& alg_params,
                double search_runtime,
                int max_trial_length,
                double eval_delta,
                int rollouts_per_mc_eval,
                int num_repeats,
                int num_threads,
                int eval_threads);

            /**
             * Returns if the env we are using is a python env
            */
            bool is_python_env();

            /**
             * Returns an instance of ThtsEnv to use for this run
            */
            std::shared_ptr<MoThtsEnv> get_env();

            /**
             * Returns and instance of ThtsManager to use for this run
            */
            std::shared_ptr<MoThtsManager> get_thts_manager(std::shared_ptr<MoThtsEnv> env);

            /**
             * Returns a root node to use for search given these params
            */
            std::shared_ptr<MoThtsDNode> get_root_search_node(
                std::shared_ptr<MoThtsEnv> env, std::shared_ptr<MoThtsManager> manager);

            /**
             * The minimum value possible in the environment
             * Useful for setting default values
             * (Note min value for each independent objective, which isn't necessarily achievable)
            */
            Eigen::ArrayXd get_env_min_value();

            /**
             * Returns the max value possible in the environment 
             * (Same as get_env_min_value, but for maximum)
            */
            Eigen::ArrayXd get_env_max_value();

            /**
             * Get max value range (max_value - min_value)
            */
            Eigen::ArrayXd get_env_value_range();
    };

    /**
     * Get a list of run id's from an experiment id
    */
    std::shared_ptr<std::vector<RunID>> get_run_ids_from_expr_id_prefix(std::string expr_id_prefix);

    /**
     * Class for running hyperparam optimisation
     * 
     * 'alg_param_ids' 
     *      is used to map between vectors (used in bayesopt) and param ids
     * 'alg_param_min_max[param_id]' 
     *      specifies the maximum and minimum values to use in bayesopt for param id
     *      N.B. min and max can be arbitrary for a boolean value, but may as well be 0.0, and 1.0
     *          and for integer value, we will sample in the *integer* range [min,max)
     */
    class HyperparamOptimiser : public bayesopt::ContinuousModel
    {
        public:
            int num_hyperparams;

            std::string env_id;
            std::string expr_id;
            std::time_t expr_timestamp;
            std::string alg_id;

            std::vector<std::string> alg_param_ids;
            std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max;

            double search_runtime;
            int max_trial_length;
            double eval_delta;
            int rollouts_per_mc_eval;
            int num_repeats;
            int num_threads;
            int eval_threads;
            int num_envs;

            double best_eval;
            std::unordered_map<std::string, double> best_alg_params;

            std::ofstream &results_fs;
            int hp_opt_iter;
            
            HyperparamOptimiser(
                std::string env_id,
                std::string expr_id,
                std::time_t expr_timestamp,
                std::string alg_id,
                std::unordered_map<std::string, std::pair<double,double>> alg_params_min_max,
                double search_runtime,
                int max_trial_length,
                double eval_delta,
                int rollouts_per_mc_eval,
                int num_repeats,
                int num_threads,
                int eval_threads,
                bayesopt::Parameters params,
                std::ofstream &results_fs);

            bool is_python_env();

            virtual std::unordered_map<std::string, double> get_alg_params_from_bayesopt_vec(bayesopt::vectord vec);

            bool get_bool_val_from_cts_sample(double sample_val, int min, int max);

            int get_int_val_from_cts_sample(double sample_val, int min, int max);

            virtual double evaluateSample(const bayesopt::vectord &query) override;

            void write_header();

        private:
            void write_eval_line(std::unordered_map<std::string,double> alg_params, double eval);

        public:
            void write_best_eval();
    };

    /**
     * Creates and returns a hyperparamters optimiser from experiment id
    */
    std::shared_ptr<HyperparamOptimiser> get_hyperparam_optimiser_from_expr_id(
        std::string expr_id, std::time_t expr_timestamp, std::ofstream &hp_opt_fs);
}