#pragma once

#include "main_aux/configs/config_map.h"


// ---------------------------------------------------------------------------------------------------------------------
// Experiment configs
// ---------------------------------------------------------------------------------------------------------------------

#include "main_aux/configs/xpr_config_100_intro.h"
#include "main_aux/configs/xpr_config_101_intro.h"

#include "main_aux/configs/xpr_config_110_dchain.h"
#include "main_aux/configs/xpr_config_111_mod_dchain.h"
#include "main_aux/configs/xpr_config_112_entropy_trap_temp.h"
#include "main_aux/configs/xpr_config_113_entropy_trap.h"

#include "main_aux/configs/xpr_config_420_frozen_lake_sparse_big_heuristic.h"
#include "main_aux/configs/xpr_config_422_frozen_lake_sparse_big_heuristic.h"
#include "main_aux/configs/xpr_config_424_frozen_lake_sparse_big_heuristic.h"
#include "main_aux/configs/xpr_config_425_frozen_lake_sparse_big_heuristic.h"

#include "main_aux/configs/xpr_config_460_frozen_lake_sparse_big_heuristic_mcts.h"
#include "main_aux/configs/xpr_config_462_frozen_lake_sparse_big_heuristic_mcts.h"

#include "main_aux/configs/xpr_config_470_frozen_lake_sparse_big_heuristic_mcts.h"
#include "main_aux/configs/xpr_config_471_frozen_lake_sparse_big_heuristic_mcts.h"
#include "main_aux/configs/xpr_config_472_frozen_lake_sparse_big_heuristic_mcts.h"

#include "main_aux/configs/xpr_config_480_slippy_frozen_lake.h"
#include "main_aux/configs/xpr_config_481_slippy_frozen_lake.h"
#include "main_aux/configs/xpr_config_482_slippy_frozen_lake.h"
#include "main_aux/configs/xpr_config_483_slippy_frozen_lake.h"

#include "main_aux/configs/xpr_config_490_sailing_north.h"
#include "main_aux/configs/xpr_config_491_sailing_north.h"
#include "main_aux/configs/xpr_config_492_sailing_north.h"

#include "main_aux/configs/xpr_config_500_sailing_south_east.h"
#include "main_aux/configs/xpr_config_501_sailing_south_east.h"
#include "main_aux/configs/xpr_config_502_sailing_south_east.h"

#include "main_aux/configs/xpr_config_700_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_701_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_710_frozen_lake_sparse_scenic_route.h"
#include "main_aux/configs/xpr_config_711_frozen_lake_sparse_scenic_route.h"

#include "main_aux/configs/xpr_config_800_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_801_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_810_frozen_lake_slippy_scenic_route.h"
#include "main_aux/configs/xpr_config_811_frozen_lake_slippy_scenic_route.h"

#include "main_aux/configs/xpr_config_900_sailing.h"
#include "main_aux/configs/xpr_config_901_sailing.h"
#include "main_aux/configs/xpr_config_910_sailing.h"
#include "main_aux/configs/xpr_config_911_sailing.h"


// ---------------------------------------------------------------------------------------------------------------------
// List of all configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::vector<std::vector<ConfigMap>> ALL_CONFIGS =
{
    CONFIG_100,
    CONFIG_101,

    CONFIG_110,
    CONFIG_111,
    CONFIG_112,
    CONFIG_113,

    CONFIG_420,
    CONFIG_422,
    CONFIG_424,
    CONFIG_425,

    CONFIG_460,
    CONFIG_462,

    CONFIG_470,
    CONFIG_471,
    CONFIG_472,

    CONFIG_480,
    CONFIG_481,
    CONFIG_482,
    CONFIG_483,

    CONFIG_490,
    CONFIG_491,
    CONFIG_492,

    CONFIG_500,
    CONFIG_501,
    CONFIG_502,

    CONFIG_700,
    CONFIG_701,
    CONFIG_710,
    CONFIG_711,

    CONFIG_800,
    CONFIG_801,
    CONFIG_810,
    CONFIG_811,

    CONFIG_900,
    CONFIG_901,
    CONFIG_910,
    CONFIG_911,
};