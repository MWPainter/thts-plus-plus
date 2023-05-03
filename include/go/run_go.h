#pragma once

#include <memory>
#include <string>
#include <unordered_map>

static const std::string PARAM_BIAS_OR_SEARCH_TEMP = "bias";
static const std::string PARAM_BIAS_OR_SEARCH_TEMP_OPP = "bias_opp";

static const std::string PARAM_MENTS_ROOT_EPS = "root_eps";
static const std::string PARAM_MENTS_ROOT_EPS_OPP = "root_eps_opp";
static const std::string PARAM_MENTS_EPS = "eps";
static const std::string PARAM_MENTS_EPS_OPP = "eps_opp";
static const std::string PARAM_PRIOR_COEFF = "prior_coeff";
static const std::string PARAM_PRIOR_COEFF_OPP = "prior_coeff_opp";

static const std::string PARAM_INIT_DECAY_TEMP = "init_decay_temp";
static const std::string PARAM_INIT_DECAY_TEMP_OPP = "init_decay_temp_opp";
static const std::string PARAM_DECAY_TEMP_VISITS_SCALE = "visits_scale";
static const std::string PARAM_DECAY_TEMP_VISITS_SCALE_OPP = "visits_scale_opp";
static const std::string PARAM_DECAY_TEMP_USE_SIGMOID = "use_sigmoid";
static const std::string PARAM_DECAY_TEMP_USE_SIGMOID_OPP = "use_sigmoid_opp";

static const std::string PARAM_USE_AVG_RETURN = "use_avg_return";
static const std::string PARAM_USE_AVG_RETURN_OPP = "use_avg_return_opp";

static const std::string ALG_ID_KATA = "kata";
static const std::string ALG_ID_UCT = "uct";
static const std::string ALG_ID_PUCT = "puct";
static const std::string ALG_ID_MENTS = "ments";
static const std::string ALG_ID_DBMENTS = "db-ments";
static const std::string ALG_ID_DENTS = "dents";
static const std::string ALG_ID_RENTS = "rents";
static const std::string ALG_ID_TENTS = "tents";
static const std::string ALG_ID_EST = "est";
static const std::string ALG_ID_UNI = "unfrm";

namespace thts {
    /**
     * Typedef to make quicker to write types
    */
    typedef std::unordered_map<std::string, double> GoAlgParams;

    /**
     * Performs all of the (replicated) runs corresponding to 'run_id'
    */
    void run_go_games(
        std::string expr_id, 
        std::string alg1_id, 
        std::string alg2_id, 
        int board_size, 
        int num_games, 
        double komi, 
        bool use_time_controls,
        double trials_or_time_per_move,
        int num_threads,
        bool use_filenames_for_hps=false,
        std::shared_ptr<GoAlgParams> alg_params=nullptr,
        std::string hps_key="",
        std::string hps_opp_key="");
}