#pragma once

#include "main_mo/configs/constants_envs.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <stdexcept>

/**
 * This file defines XPR_PARAM_ID constants
 * - XPR_PARAM_ID = id's for parameters that each experiment should specify
 */


// ---------------------------------------------------------------------------------------------------------------------
// Tags to indicate what a config dictionary contains to indicate experiment level params in configs
// ---------------------------------------------------------------------------------------------------------------------

static const std::string XPR_OR_ALG_ID_TAG = "xpr_or_alg_id";                           // Special key to indicate if a dictonray is specifying xpr level params or alg level params
static const std::string XPR_PARAMS_ID_TAG = "xpr_params";                              // Special value to indicate dictionary is specifying xpr level params

// ---------------------------------------------------------------------------------------------------------------------
// Constants used in the tree search, but to be varied on a per experiment basis
// ---------------------------------------------------------------------------------------------------------------------

static const std::string XPR_PARAM_ID_NAME = "xpr_name";                                // user readable name for experiment
static const std::string XPR_PARAM_ID_ENV = "env_id";                                   // the env id for this experiment
static const std::string XPR_PARAM_ID_ENV_SIZE = "env_size";                            // the size of the environment (if applicable)
static const std::string XPR_PARAM_ID_MCTS_MODE = "mcts_mode";                          // if MCTS mode should be used
static const std::string XPR_PARAM_ID_GRAPH_SEARCH = "graph_search";                    // if should run over graph instead of tree (transposition table use)
static const std::string XPR_PARAM_ID_VECTOR_VISIT_COUNTS = "vec_visit_counts";         // whether to use vector visit counts
static const std::string XPR_PARAM_ID_MAX_TRIAL_LENGTH = "max_trial_length";            // max trial length
static const std::string XPR_PARAM_ID_RUNTIME_BOUNDED = "runtime_bounded";              // if algorithms should be bounded using runtime (or number of trials)
static const std::string XPR_PARAM_ID_TERMINATION_BOUND = "term_bound";                 // runtime (or #trials) that algorithm is allowed
static const std::string XPR_PARAM_ID_REPEATED_RUNS_PER_ALG = "num_repeats";            // number of times to repeat running each algorithms
static const std::string XPR_PARAM_ID_SEARCH_THREADS = "search_threads";                // number of threads to use in search
static const std::string XPR_PARAM_ID_EVAL_DELTA = "eval_delta";                        // delta (in runtime/#trials) to evaluate algorithms at
static const std::string XPR_PARAM_ID_EVAL_ROLLOUTS = "eval_rollouts";                  // numbber of rollouts for each MC eval
static const std::string XPR_PARAM_ID_EVAL_THREADS = "eval_threads";                    // number of threads to use in evaluation
static const std::string XPR_PARAM_ID_CONVEX_HULL_MAX_SIZE = "convex_hull_max_size";    // max size of convex hull
static const std::string XPR_PARAM_ID_CONVEX_HULL_TOLERANCE = "convex_hull_tolerance";  // tolerance controlling how close to consider points to be "equal" for convex hull
static const std::string XPR_PARAM_ID_USE_SOLVED_LABELLING = "use_solved_labelling";    // whether to use solved labelling
static const std::string XPR_PARAM_ID_SOLVED_LABELLING_FAIL_CONFIDENCE = "solved_labelling_fail_confidence";    // confidence that confidence interval fails to contain true solved value
static const std::string XPR_PARAM_ID_SOLVED_LABELLING_TOLERANCE = "solved_labelling_tolerance";    // tolerance for when solved values when consider a node solved

static const std::string XPR_PARAM_ID_SM_PUSH_RADIUS = "sm_push_radius";                // simplex maps: push radius
static const std::string XPR_PARAM_ID_SM_MAX_NEIGHBOURS_TO_PUSH_TO = "sm_max_neighbours_to_push_to"; // simplex maps: max neighbours to push to
static const std::string XPR_PARAM_ID_SM_MIN_SIMPLEX_RADIUS = "sm_min_simplex_radius";    // simplex maps: min simplex radius
static const std::string XPR_PARAM_ID_SM_SIMPLEX_SPLIT_COUNTER_THRESHOLD = "sm_simplex_split_counter_threshold"; // simplex maps: simplex split counter threshold
static const std::string XPR_PARAM_ID_SM_USE_APPROX_NEAREST_VERTEX = "sm_use_approx_nearest_vertex"; // simplex maps: use approx nearest vertex
static const std::string XPR_PARAM_ID_SM_EVENTUALLY_CONFORMING_SIMPLEX_MAP = "sm_eventually_conforming_simplex_map"; // simplex maps: eventually conforming simplex map
static const std::string XPR_PARAM_ID_SM_ALWAYS_ALLOW_NON_CONFORMING_SIMPLEX_TO_SPLIT = "sm_always_allow_non_conforming_simplex_to_split"; // simplex maps: always allow non conforming simplex to split


static const int NO_ENV_SIZE = -1;

