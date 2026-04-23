#pragma once

#include "main_mo/configs/config_map.h"


// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_mo/configs/xpr_config_000_debug.h"
// #include "main_mo/configs/xpr_config_001_debug.h"
// #include "main_mo/configs/xpr_config_002_debug.h"
// #include "main_mo/configs/xpr_config_003_debug.h"
// #include "main_mo/configs/xpr_config_004_debug.h"
// #include "main_mo/configs/xpr_config_005_debug.h"
// #include "main_mo/configs/xpr_config_006_debug.h"
// #include "main_mo/configs/xpr_config_007_debug.h"
// #include "main_mo/configs/xpr_config_008_debug.h"
// #include "main_mo/configs/xpr_config_009_debug.h"
// #include "main_mo/configs/xpr_config_010_debug.h"
// #include "main_mo/configs/xpr_config_011_debug.h"
// #include "main_mo/configs/xpr_config_012_debug.h"
// #include "main_mo/configs/xpr_config_013_debug.h"
// #include "main_mo/configs/xpr_config_014_debug.h"
// #include "main_mo/configs/xpr_config_015_debug.h"
// #include "main_mo/configs/xpr_config_016_debug.h"
// #include "main_mo/configs/xpr_config_020_debug.h"
// #include "main_mo/configs/xpr_config_021_debug.h"
// #include "main_mo/configs/xpr_config_030_debug.h"
// #include "main_mo/configs/xpr_config_031_debug.h"
// #include "main_mo/configs/xpr_config_040_debug.h"
// #include "main_mo/configs/xpr_config_041_debug.h"
#include "main_mo/configs/xpr_config_050_debug.h"
#include "main_mo/configs/xpr_config_051_debug.h"
#include "main_mo/configs/xpr_config_052_debug.h"
#include "main_mo/configs/xpr_config_053_debug.h"
#include "main_mo/configs/xpr_config_060_debug.h"
#include "main_mo/configs/xpr_config_061_debug.h"
#include "main_mo/configs/xpr_config_062_debug.h"
#include "main_mo/configs/xpr_config_070_debug.h"
#include "main_mo/configs/xpr_config_071_debug.h"
#include "main_mo/configs/xpr_config_072_debug.h"

#include "main_mo/configs/xpr_config_400_dst.h"
#include "main_mo/configs/xpr_config_400a_dst.h"
#include "main_mo/configs/xpr_config_401_debug.h"
#include "main_mo/configs/xpr_config_410_dst_stoch.h"
#include "main_mo/configs/xpr_config_410a_dst_stoch.h"
#include "main_mo/configs/xpr_config_411_debug.h"
#include "main_mo/configs/xpr_config_420_dst_stoch_clm.h"
#include "main_mo/configs/xpr_config_420a_dst_stoch_clm.h"

#include "main_mo/configs/xpr_config_440_dst_improved.h"
#include "main_mo/configs/xpr_config_450_dst_improved_stoch.h"
#include "main_mo/configs/xpr_config_460_dst_improved_stoch_clm.h"

#include "main_mo/configs/xpr_config_500_fruit_tree.h"
#include "main_mo/configs/xpr_config_510_fruit_tree_stoch.h"
#include "main_mo/configs/xpr_config_520_resource_gather.h"
#include "main_mo/configs/xpr_config_530_resource_gather_timed.h"
#include "main_mo/configs/xpr_config_540_breakable_bottles.h"
#include "main_mo/configs/xpr_config_550_four_room.h"
#include "main_mo/configs/xpr_config_560_four_room_timed.h"
#include "main_mo/configs/xpr_config_580_resource_gather_cpp.h"
#include "main_mo/configs/xpr_config_590_resource_gather_timed_cpp.h"

#include "main_mo/configs/xpr_config_700_dst.h"
#include "main_mo/configs/xpr_config_700a_dst.h"
#include "main_mo/configs/xpr_config_700b_dst.h"
#include "main_mo/configs/xpr_config_710_dst_stoch.h"
#include "main_mo/configs/xpr_config_710a_dst_stoch.h"
#include "main_mo/configs/xpr_config_710b_dst_stoch.h"
#include "main_mo/configs/xpr_config_720_dst_stoch_clm.h"
#include "main_mo/configs/xpr_config_720a_dst_stoch_clm.h"
#include "main_mo/configs/xpr_config_720b_dst_stoch_clm.h"

#include "main_mo/configs/xpr_config_800_dim_scaling.h"

// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<ConfigMap>> ALL_CONFIGS =
{
    CONFIG_000,
    // CONFIG_001,
    // CONFIG_002,
    // CONFIG_003,
    // CONFIG_004,
    // CONFIG_005,
    // CONFIG_006,
    // CONFIG_007,
    // CONFIG_008,
    // CONFIG_009,
    // CONFIG_010,
    // CONFIG_011,
    // CONFIG_012,
    // CONFIG_013,
    // CONFIG_014,
    // CONFIG_015,
    // CONFIG_016,
    // CONFIG_020,
    // CONFIG_021,
    // CONFIG_030,
    // CONFIG_031,
    // CONFIG_040,
    // CONFIG_041,
    CONFIG_050,
    CONFIG_051,
    CONFIG_052,
    CONFIG_053,
    CONFIG_060,
    CONFIG_061,
    CONFIG_062,
    CONFIG_070,
    CONFIG_071,
    CONFIG_072,

    CONFIG_400,
    CONFIG_400a,
    CONFIG_401,
    CONFIG_410,
    CONFIG_410a,
    CONFIG_411,
    CONFIG_420,
    CONFIG_420a,
    
    CONFIG_440,
    CONFIG_450,
    CONFIG_460,

    CONFIG_500,
    CONFIG_510,
    CONFIG_520,
    CONFIG_530,
    CONFIG_540,
    CONFIG_550,
    CONFIG_560,
    CONFIG_580,
    CONFIG_590,
    
    CONFIG_700,
    CONFIG_700a,
    CONFIG_700b,
    CONFIG_710,
    CONFIG_710a,
    CONFIG_710b,
    CONFIG_720,
    CONFIG_720a,
    CONFIG_720b,

    CONFIG_800,
};