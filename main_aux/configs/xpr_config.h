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
};