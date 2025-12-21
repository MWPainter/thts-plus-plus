#pragma once

#include "main_aux/configs/config_map.h"


// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_aux/configs/xpr_config_000_debug.h"
#include "main_aux/configs/xpr_config_100_dchain_temp.h"
#include "main_aux/configs/xpr_config_101_mod_dchain_temp.h"
#include "main_aux/configs/xpr_config_102_entropy_trap_temp.h"
#include "main_aux/configs/xpr_config_103_entropy_trap_15_temp.h"
#include "main_aux/configs/xpr_config_400_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_401_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_402_frozen_lake_dense.h"
#include "main_aux/configs/xpr_config_410_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_411_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_412_frozen_lake_sparse.h"
// #include "main_aux/configs/xpr_config_420_slippy_frozen_lake_dense.h"
// #include "main_aux/configs/xpr_config_421_slippy_frozen_lake_dense.h"
// #include "main_aux/configs/xpr_config_422_slippy_frozen_lake_dense.h"
// #include "main_aux/configs/xpr_config_430_slippy_frozen_lake_sparse.h"
// #include "main_aux/configs/xpr_config_431_slippy_frozen_lake_sparse.h"
// #include "main_aux/configs/xpr_config_432_slippy_frozen_lake_sparse.h"
#include "main_aux/configs/xpr_config_440_sailing_north.h"
#include "main_aux/configs/xpr_config_441_sailing_north.h"
#include "main_aux/configs/xpr_config_442_sailing_north.h"
#include "main_aux/configs/xpr_config_450_sailing_south_east.h"
#include "main_aux/configs/xpr_config_451_sailing_south_east.h"
#include "main_aux/configs/xpr_config_452_sailing_south_east.h"


// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<ConfigMap>> ALL_CONFIGS =
{
    CONFIG_000,
    CONFIG_100,
    CONFIG_101,
    CONFIG_102,
    CONFIG_103,

    CONFIG_400,
    CONFIG_401,
    CONFIG_402,
    CONFIG_410,
    CONFIG_411,
    CONFIG_412,
    // CONFIG_420,
    // CONFIG_421,
    // CONFIG_422,
    // CONFIG_430,
    // CONFIG_431,
    // CONFIG_432,
    CONFIG_440,
    CONFIG_441,
    CONFIG_442,
    CONFIG_450,
    CONFIG_451,
    CONFIG_452,
};