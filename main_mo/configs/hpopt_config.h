#pragma once

#include "main_mo/configs/hpopt_config_map.h"

// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_mo/configs/hpopt_config_001_debug.h"
#include "main_mo/configs/hpopt_config_100_dst.h"
#include "main_mo/configs/hpopt_config_110_dst_stoch.h"
#include "main_mo/configs/hpopt_config_120_dst_gym.h"
#include "main_mo/configs/hpopt_config_130_dst_gym_stoch.h"
#include "main_mo/configs/hpopt_config_140_dst_improved.h"
#include "main_mo/configs/hpopt_config_150_dst_improved_stoch.h"
#include "main_mo/configs/hpopt_config_200_fruit_tree.h"
#include "main_mo/configs/hpopt_config_210_fruit_tree_stoch.h"
#include "main_mo/configs/hpopt_config_220_resource_gather.h"
#include "main_mo/configs/hpopt_config_230_resource_gather_timed.h"
#include "main_mo/configs/hpopt_config_240_breakable_bottles.h"
#include "main_mo/configs/hpopt_config_250_four_room.h"
#include "main_mo/configs/hpopt_config_260_four_room_timed.h"
#include "main_mo/configs/hpopt_config_270_minecart.h"
#include "main_mo/configs/hpopt_config_280_resource_gather_cpp.h"
#include "main_mo/configs/hpopt_config_290_resource_gather_timed_cpp.h"

// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<HpoptConfigMap>> ALL_HPOPT_CONFIGS =
{
    HPOPT_CONFIG_001,

    HPOPT_CONFIG_100,
    HPOPT_CONFIG_110,
    HPOPT_CONFIG_120,
    HPOPT_CONFIG_130,
    HPOPT_CONFIG_140,
    HPOPT_CONFIG_150,
    
    HPOPT_CONFIG_200,
    HPOPT_CONFIG_210,
    HPOPT_CONFIG_220,
    HPOPT_CONFIG_230,
    HPOPT_CONFIG_240,
    HPOPT_CONFIG_250,
    HPOPT_CONFIG_260,
    HPOPT_CONFIG_270,
    HPOPT_CONFIG_280,
    HPOPT_CONFIG_290,
};