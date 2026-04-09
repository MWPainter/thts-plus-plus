#pragma once

#include "main_aux/configs/hpopt_config_map.h"

// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

// #include "main_aux/configs/hpopt_config_001_debug.h"

#include "main_aux/configs/hpopt_config_300_frozen_lake_det_dense.h"
#include "main_aux/configs/hpopt_config_310_frozen_lake_det_sparse.h"
#include "main_aux/configs/hpopt_config_320_frozen_lake_det_sparse_big.h"
#include "main_aux/configs/hpopt_config_321_frozen_lake_det_sparse_big.h"
#include "main_aux/configs/hpopt_config_330_frozen_lake_slippy.h"
#include "main_aux/configs/hpopt_config_331_frozen_lake_slippy.h"
#include "main_aux/configs/hpopt_config_340_sailing_north.h"
#include "main_aux/configs/hpopt_config_350_sailing_south_east.h"
#include "main_aux/configs/hpopt_config_351_sailing_south_east.h"
#include "main_aux/configs/hpopt_config_360_frozen_lake_det_sparse.h"
#include "main_aux/configs/hpopt_config_370_frozen_lake_det_sparse.h"

// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<HpoptConfigMap>> ALL_HPOPT_CONFIGS =
{
    // HPOPT_CONFIG_001,
    HPOPT_CONFIG_300,
    HPOPT_CONFIG_310,
    HPOPT_CONFIG_320,
    HPOPT_CONFIG_321,
    HPOPT_CONFIG_330,
    HPOPT_CONFIG_331,
    HPOPT_CONFIG_340,
    HPOPT_CONFIG_350,
    HPOPT_CONFIG_351,
    HPOPT_CONFIG_360,
    HPOPT_CONFIG_370,
};