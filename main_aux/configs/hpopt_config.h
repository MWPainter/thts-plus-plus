#pragma once

#include "main_aux/configs/hpopt_config_map.h"

// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_aux/configs/hpopt_config_001_debug.h"

#include "main_aux/configs/hpopt_config_300_frozen_lake_det_dense.h"
#include "main_aux/configs/hpopt_config_301_frozen_lake_det_dense.h"
#include "main_aux/configs/hpopt_config_310_frozen_lake_det_sparse.h"
#include "main_aux/configs/hpopt_config_320_frozen_lake_stoch_dense.h"
#include "main_aux/configs/hpopt_config_330_frozen_lake_stoch_sparse.h"
#include "main_aux/configs/hpopt_config_340_sailing_north.h"
#include "main_aux/configs/hpopt_config_350_sailing_south_east.h"

// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<HpoptConfigMap>> ALL_HPOPT_CONFIGS =
{
    HPOPT_CONFIG_001,
    HPOPT_CONFIG_300,
    HPOPT_CONFIG_301,
    HPOPT_CONFIG_310,
    // HPOPT_CONFIG_320,
    HPOPT_CONFIG_330,
    HPOPT_CONFIG_340,
    HPOPT_CONFIG_350,
};