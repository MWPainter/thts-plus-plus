#pragma once

#include <string>
#include <unordered_set>

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - d_chain
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_D_CHAIN_10 = "dchain(D=10,R=1.0)";
static const std::string ENV_ID_MOD_D_CHAIN_10 = "dchain(D=10,R=0.5)";
static const std::string ENV_ID_ENTROPY_TRAP_10 = "entropy_trap(D=10,H=10)";
static const std::string ENV_ID_ENTROPY_TRAP_15 = "entropy_trap(D=15,H=15)";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - frozen lake (comparing rewards)
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_FROZEN_LAKE_NO_HOLE_DENSE = "frozen_lake_no_hole_dense";
static const std::string ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_LEN = "frozen_lake_no_hole_sparse_len";
static const std::string ENV_ID_FROZEN_LAKE_NO_HOLE_SPARSE_DISCOUNTED = "frozen_lake_no_hole_sparse_discounted";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - frozen lake (deterministic)
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_FROZEN_LAKE_D_8x8 = "frozen_lake_(map=8x8,dense)";
static const std::string ENV_ID_FROZEN_LAKE_S_8x8 = "frozen_lake_(map=8x8,sparse)";
static const std::string ENV_ID_FROZEN_LAKE_D_8x16 = "frozen_lake_(map=8x16,dense)";
static const std::string ENV_ID_FROZEN_LAKE_S_8x16 = "frozen_lake_(map=8x16,sparse)";
static const std::string ENV_ID_FROZEN_LAKE_D_16x16 = "frozen_lake_(map=16x16,dense)";
static const std::string ENV_ID_FROZEN_LAKE_S_16x16 = "frozen_lake_(map=16x16,sparse)";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - frozen lake (stochastic)
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_D_4x4 = "slippy_frozen_lake_(map=4x4,dense)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_S_4x4 = "slippy_frozen_lake_(map=4x4,sparse)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_D_5x5 = "slippy_frozen_lake_(map=5x5,dense)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_S_5x5 = "slippy_frozen_lake_(map=5x5,sparse)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_D_6x6 = "slippy_frozen_lake_(map=6x6,dense)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_S_6x6 = "slippy_frozen_lake_(map=6x6,sparse)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_D_4x8 = "slippy_frozen_lake_(map=4x8,dense)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_S_4x8 = "slippy_frozen_lake_(map=4x8,sparse)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_D_4x12 = "slippy_frozen_lake_(map=4x12,dense)";
static const std::string ENV_ID_SLIPPY_FROZEN_LAKE_S_4x12 = "slippy_frozen_lake_(map=4x12,sparse)";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - sailing
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_SAILING_NORTH_ID = "sailing_(8x8,N)";
static const std::string ENV_ID_SAILING_SOUTH_EAST_ID = "sailing_(8x8,SE)";
static const std::string ENV_ID_SAILING_8x16_NORTH_ID = "sailing_(8x16,N)";
static const std::string ENV_ID_SAILING_8x16_SOUTH_EAST_ID = "sailing_(8x16,SE)";
static const std::string ENV_ID_SAILING_16x16_NORTH_ID = "sailing_(16x16,N)";
static const std::string ENV_ID_SAILING_16x16_SOUTH_EAST_ID = "sailing_(16x16,SE)";

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - python
// ---------------------------------------------------------------------------------------------------------------------

static const std::unordered_set<std::string> PY_ENVS =
{
};

// ---------------------------------------------------------------------------------------------------------------------
// Environment id's - gymnasium
// ---------------------------------------------------------------------------------------------------------------------

static const std::string ENV_ID_TAXI_GYM = "Taxi-v3"; // https://gymnasium.farama.org/environments/toy_text/taxi/

static const std::unordered_set<std::string> GYM_ENVS =
{
    ENV_ID_TAXI_GYM,
};

