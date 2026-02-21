#pragma once

#include "main_aux/configs/config_map.h"


// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_aux/configs/xpr_config_000_debug.h"
#include "main_aux/configs/xpr_config_001_debug.h"
#include "main_aux/configs/xpr_config_002_debug.h"
#include "main_aux/configs/xpr_config_003_debug.h"
#include "main_aux/configs/xpr_config_100_dchain_temp.h"
#include "main_aux/configs/xpr_config_101_mod_dchain_temp.h"
#include "main_aux/configs/xpr_config_102_entropy_trap_temp.h"
#include "main_aux/configs/xpr_config_102a_entropy_trap_temp.h"
#include "main_aux/configs/xpr_config_102b_entropy_trap_temp.h"
#include "main_aux/configs/xpr_config_103_entropy_trap_15_temp.h"
#include "main_aux/configs/xpr_config_400_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401a_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401b_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401c_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401d_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401e_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_402_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_410_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_411_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_412_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_430_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_431_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_432_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_433_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_434_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_440_sailing_north.h"
#include "main_aux/configs/xpr_config_441_sailing_north.h"
#include "main_aux/configs/xpr_config_442_sailing_north.h"
#include "main_aux/configs/xpr_config_450_sailing_south_east.h"
#include "main_aux/configs/xpr_config_451_sailing_south_east.h"
#include "main_aux/configs/xpr_config_452_sailing_south_east.h"
#include "main_aux/configs/xpr_config_600_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_601_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_602_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_610_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_611_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_612_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_620_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_621_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_622_frozen_lake_dense_scenic_route.h"
#include "main_aux/configs/xpr_config_700_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_701_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_702_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_710_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_711_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_712_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_720_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_721_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_722_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_800_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_801_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_802_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_810_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_811_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_812_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_820_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_821_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_822_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_999_ments_entropy_trap_debug.h"


// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<ConfigMap>> ALL_CONFIGS =
{
    CONFIG_000,
    CONFIG_001,
    CONFIG_002,
    CONFIG_003,
    
    CONFIG_100,
    CONFIG_101,
    CONFIG_102,
    CONFIG_102a,
    CONFIG_102b,
    CONFIG_103,

    CONFIG_400,
    CONFIG_401,
    CONFIG_402,

    CONFIG_401a,
    CONFIG_401b,
    CONFIG_401c,
    CONFIG_401d,
    CONFIG_401e,

    CONFIG_410,
    CONFIG_411,
    CONFIG_412,

    CONFIG_430,
    CONFIG_431,
    CONFIG_432,
    CONFIG_433,
    CONFIG_434,

    CONFIG_440,
    CONFIG_441,
    CONFIG_442,

    CONFIG_450,
    CONFIG_451,
    CONFIG_452,

    CONFIG_600,
    CONFIG_601,
    CONFIG_602,

    CONFIG_610,
    CONFIG_611,
    CONFIG_612,

    CONFIG_620,
    CONFIG_621,
    CONFIG_622,

    CONFIG_700,
    CONFIG_701,
    CONFIG_702,

    CONFIG_710,
    CONFIG_711,
    CONFIG_712,

    CONFIG_720,
    CONFIG_721,
    CONFIG_722,

    CONFIG_800,
    CONFIG_801,
    CONFIG_802,

    CONFIG_810,
    CONFIG_811,
    CONFIG_812,

    CONFIG_820,
    CONFIG_821,
    CONFIG_822,

    CONFIG_999,
};