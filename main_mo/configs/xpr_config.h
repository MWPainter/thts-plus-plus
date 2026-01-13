#pragma once

#include "main_mo/configs/config_map.h"


// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_mo/configs/xpr_config_000_debug.h"
#include "main_mo/configs/xpr_config_001_debug.h"
#include "main_mo/configs/xpr_config_002_debug.h"
#include "main_mo/configs/xpr_config_003_debug.h"
#include "main_mo/configs/xpr_config_004_debug.h"


// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<ConfigMap>> ALL_CONFIGS =
{
    CONFIG_000,
    CONFIG_001,
    CONFIG_002,
    CONFIG_003,
    CONFIG_004,
};