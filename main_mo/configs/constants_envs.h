#pragma once

#include <string>
#include <unordered_set>

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - debug envs
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_DEBUG_1 = "debug_env_1"; // not stoch + 2 rew
static const std::string ENV_ID_DEBUG_2 = "debug_env_2"; // stoch + 2 rew
static const std::string ENV_ID_DEBUG_3 = "debug_env_3"; // not stoch + 4 rew
static const std::string ENV_ID_DEBUG_4 = "debug_env_4"; // stoch + 4 rew

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - debug envs (python)
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_PY_DEBUG_1 = "py_debug_env_1"; // not stoch + 2 rew
static const std::string ENV_ID_PY_DEBUG_2 = "py_debug_env_2"; // stoch + 2 rew
static const std::string ENV_ID_PY_DEBUG_3 = "py_debug_env_3"; // not stoch + 4 rew
static const std::string ENV_ID_PY_DEBUG_4 = "py_debug_env_4"; // stoch + 4 rew

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - DST (vamplew = original)
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_IMPROVED_DST = "deep-sea-treasure-improved";
static const std::string ENV_ID_IMPROVED_STOCH_DST = "deep-sea-treasure-improved-stoch";
static const std::string ENV_ID_VAMPLEW_DST = "deep-sea-treasure-vamplew";
static const std::string ENV_ID_VAMPLEW_STOCH_DST = "deep-sea-treasure-vamplew-stoch";
// TODO: variable sized DST for scalability
// TODO: make sure that can input a map that is randomly generated + cache the generated ones like in frozen lake
 
// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - fruit tree
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_FRUIT_TREE_7 = "fruit-tree-depth-7";
static const std::string ENV_ID_FRUIT_TREE_STOCH_5 = "fruit-tree-stoch-depth-5";
static const std::string ENV_ID_FRUIT_TREE_STOCH_7 = "fruit-tree-stoch-depth-7";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - mo gymnasium (discr obs / discr act)
// ---------------------------------------------------------------------------------------------------------------------

// static const std::string ENV_ID_DST = "deep-sea-treasure-v0";                   // unused (using improved DST vamplew wrapper)            
// static const std::string ENV_ID_DST_CONC = "deep-sea-treasure-concave-v0";      // unused (using improved DST vamplew wrapper)
// static const std::string ENV_ID_DST_MIRR = "deep-sea-treasure-mirrored-v0";     // unused (using improved DST vamplew wrapper)
static const std::string ENV_ID_RESOURCE_GATHER = "resource-gathering-v0";
static const std::string ENV_ID_BREAKABLE_BOTTLES = "breakable-bottles-v0";
// static const std::string ENV_ID_FRUIT_TREE = "fruit-tree-v0";                   // unused (using custom copy with opt for stoch)
static const std::string ENV_ID_FOUR_ROOM = "four-room-v0";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - mo gymnasium (cts obs / discr act)
// ---------------------------------------------------------------------------------------------------------------------

// TODO: which of these have constant initial state?
// static const std::string ENV_ID_MOUNTAIN_CAR = "mo-mountaincar-v0";             
// static const std::string ENV_ID_LUNAR_LANDER = "mo-lunar-lander-v2";         
// static const std::string ENV_ID_MINECART_DETERMINISTIC = "minecart-deterministic-v0";
// static const std::string ENV_ID_MINECART = "minecart-v0";
// static const std::string ENV_ID_HIGHWAY = "mo-highway-v0";  

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - mo gymnasium (discr obs / discr act) + extra time cost
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_RESOURCE_GATHER_TIMED = "resource-gathering-timed-v0";
static const std::string ENV_ID_FOUR_ROOM_TIMED = "four-room-timed-v0";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - python
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> PY_ENVS =
{
    ENV_ID_IMPROVED_DST,
    ENV_ID_IMPROVED_STOCH_DST,
    ENV_ID_VAMPLEW_DST,
    ENV_ID_VAMPLEW_STOCH_DST,

    ENV_ID_FRUIT_TREE_7,
    ENV_ID_FRUIT_TREE_STOCH_5,
    ENV_ID_FRUIT_TREE_STOCH_7,
};

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - gymnasium
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> GYM_ENVS =
{
    ENV_ID_RESOURCE_GATHER,
    ENV_ID_BREAKABLE_BOTTLES,
    // ENV_ID_FRUIT_TREE,
    ENV_ID_FOUR_ROOM,
    
    // ENV_ID_MOUNTAIN_CAR,
    // ENV_ID_LUNAR_LANDER,
    // ENV_ID_MINECART_DETERMINISTIC,
    // ENV_ID_MINECART, 
    // ENV_ID_HIGHWAY,
};

