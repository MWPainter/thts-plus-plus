#include "go/run_go.h"

#include "KataGo/cpp/game/board.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// 000 series - old tuning on 9x9
static const std::string EXPR_ID_DEBUG = "000_debug";

// Old tests keeping to remind ourselves we did it
static const std::string EXPR_ID_KOMI = "001_komi_9x9";
static const std::string EXPR_ID_KATA_RECOMMEND_TEST = "008_test_kata_recommend";
static const std::string EXPR_ID_KATA_THREAD_TEST = "011_kata_thread_test";
static const std::string EXPR_ID_KATA_THREAD_TEST_WITH_DIRICHLET = "011a_kata_thread_test_with_dirichlet";
static const std::string EXPR_ID_EST_THREAD_TEST = "012_est_thread_test";
static const std::string EXPR_ID_DIRICHLET_NOISE = "013_dirichlet_noise";

// w000 series - tuning dynamic programming algorithms
static const std::string EXPR_ID_W000_BTS_TEMP_CONST = "w000_bts_tune_temp_no_decay";
static const std::string EXPR_ID_W001_BTS_TEMP_SQRT_DECAY = "w001_bts_tune_temp_sqrt_decay";
static const std::string EXPR_ID_W002_BTS_TEMP_LOG_DECAY = "w002_bts_tune_temp_log_decay";
static const std::string EXPR_ID_W003_BTS_TEMP_COMPARE_DECAY = "w003_bts_tune_temp_compare";
static const std::string EXPR_ID_W010_BTS_MOST_VISITED_TEMP_CONST = "w010_bts_tune_temp_most_visited_no_decay";
static const std::string EXPR_ID_W011_BTS_MOST_VISITED_TEMP_SQRT_DECAY = "w011_bts_tune_temp_most_visited_sqrt_decay";
static const std::string EXPR_ID_W012_BTS_MOST_VISITED_TEMP_LOG = "w012_bts_tune_temp_most_visited_log_decay";
static const std::string EXPR_ID_W013_BTS_MOST_VISITED_TEMP_LOG = "w013_bts_tune_temp_most_visited_compare_decay";
static const std::string EXPR_ID_W014_BTS_COMPARE_RECOMMEND = "w014_bts_tune_compare_recommendations";
static const std::string EXPR_ID_W020_BTS_PRIOR_COEFF = "w020_bts_tune_prior_coeff";
static const std::string EXPR_ID_W021_BTS_PRIOR_COEFF = "w021_bts_tune_prior_coeff";
static const std::string EXPR_ID_W030_BTS_EPS_COEFF = "w030_bts_tune_eps_coeff";
static const std::string EXPR_ID_W040_MENTS_TEMP = "w040_ments_tune_temp";
static const std::string EXPR_ID_W041_MENTS_TEMP = "w041_ments_tune_temp_most_visit";
static const std::string EXPR_ID_W042_MENTS_TEMP = "w042_ments_tune_temp_compare_recommend";
static const std::string EXPR_ID_W050_RENTS_TEMP = "w050_rents_tune_temp";
static const std::string EXPR_ID_W051_RENTS_TEMP = "w051_rents_tune_temp_most_visit";
static const std::string EXPR_ID_W052_RENTS_TEMP = "w052_rents_tune_temp_compare_recommend";
static const std::string EXPR_ID_W060_TENTS_TEMP = "w060_tents_tune_temp";
static const std::string EXPR_ID_W061_TENTS_TEMP = "w061_tents_tune_temp_most_visit";
static const std::string EXPR_ID_W062_TENTS_TEMP = "w062_tents_tune_temp_compare_recommend";
static const std::string EXPR_ID_W070_DENTS_VALUE_TEMP = "w070_dents_tune_value_temp";

// x000 series - tuning average returns algorithms
static const std::string EXPR_ID_X000_BTS_TEMP_CONST = "x000_bts_tune_temp_no_decay";
static const std::string EXPR_ID_X001_BTS_TEMP_SQRT_DECAY = "x001_bts_tune_temp_sqrt_decay";
static const std::string EXPR_ID_X002_BTS_TEMP_LOG_DECAY = "x002_bts_tune_temp_log_decay";
static const std::string EXPR_ID_X003_BTS_TEMP_COMPARE_DECAY = "x003_bts_tune_temp_compare";
static const std::string EXPR_ID_X010_BTS_MOST_VISITED_TEMP_CONST = "x010_bts_tune_temp_most_visited_no_decay";
static const std::string EXPR_ID_X011_BTS_MOST_VISITED_TEMP_SQRT_DECAY = "x011_bts_tune_temp_most_visited_sqrt_decay";
static const std::string EXPR_ID_X012_BTS_MOST_VISITED_TEMP_LOG = "x012_bts_tune_temp_most_visited_log_decay";
static const std::string EXPR_ID_X013_BTS_MOST_VISITED_TEMP_LOG = "x013_bts_tune_temp_most_visited_compare_decay";
static const std::string EXPR_ID_X014_BTS_COMPARE_RECOMMEND = "x014_bts_tune_compare_recommendations";
static const std::string EXPR_ID_X020_BTS_PRIOR_COEFF = "x020_bts_tune_prior_coeff";
static const std::string EXPR_ID_X030_BTS_EPS_COEFF = "x030_bts_tune_eps_coeff";
static const std::string EXPR_ID_X040_MENTS_TEMP = "x040_ments_tune_temp";
static const std::string EXPR_ID_X041_MENTS_TEMP = "x041_ments_tune_temp_most_visit";
static const std::string EXPR_ID_X042_MENTS_TEMP = "x042_ments_tune_temp_compare_recommend";
static const std::string EXPR_ID_X050_RENTS_TEMP = "x050_rents_tune_temp";
static const std::string EXPR_ID_X051_RENTS_TEMP = "x051_rents_tune_temp_most_visit";
static const std::string EXPR_ID_X052_RENTS_TEMP = "x052_rents_tune_temp_compare_recommend";
static const std::string EXPR_ID_X060_TENTS_TEMP = "x060_tents_tune_temp";
static const std::string EXPR_ID_X061_TENTS_TEMP = "x061_tents_tune_temp_most_visit";
static const std::string EXPR_ID_X062_TENTS_TEMP = "x062_tents_tune_temp_compare_recommend";
static const std::string EXPR_ID_X070_DENTS_VALUE_TEMP = "x070_dents_tune_value_temp";

// y000 series - evaluating on 9x9 go
static const std::string EXPR_ID_Y000_PUCT_VS_KATA_NATIVE = "y000_puct_vs_kata_native";
static const std::string EXPR_ID_Y001_AR_BTS_VS_KATA_NATIVE = "y001_ar_bts_vs_kata_native";
static const std::string EXPR_ID_Y002_DP_BTS_VS_KATA_NATIVE = "y002_dp_bts_vs_kata_native";
static const std::string EXPR_ID_Y010_AR_VS_DP = "y010_ar_vs_dp";
static const std::string EXPR_ID_Y020_AR_RR_WITH_RAND = "y020_ar_round_robin_with_rand";
static const std::string EXPR_ID_Y021_DP_RR_WITH_RAND = "y021_dp_round_robin_with_rand";
static const std::string EXPR_ID_Y030_AR_RR = "y030_ar_round_robin";
static const std::string EXPR_ID_Y031_DP_RR = "y031_dp_round_robin";

// z000 series - evaluating on 19x19 go
static const std::string EXPR_ID_Z000_PUCT_VS_KATA_NATIVE = "z000_puct_vs_kata_native";
static const std::string EXPR_ID_Z001_AR_BTS_VS_KATA_NATIVE = "z001_ar_bts_vs_kata_native";
static const std::string EXPR_ID_Z002_DP_BTS_VS_KATA_NATIVE = "z002_dp_bts_vs_kata_native";
static const std::string EXPR_ID_Z010_AR_VS_DP = "z010_ar_vs_dp";
static const std::string EXPR_ID_Z020_AR_RR_WITH_RAND = "z020_ar_round_robin_with_rand";
static const std::string EXPR_ID_Z021_DP_RR_WITH_RAND = "z021_dp_round_robin_with_rand";
static const std::string EXPR_ID_Z030_AR_RR = "z030_ar_round_robin";
static const std::string EXPR_ID_Z031_DP_RR = "z031_dp_round_robin";

int main(int argc, char* argv[]) {
    if (argc < 2) {
        throw runtime_error("Expecting command line args");
    }

    // Katago needs a global init for hashing
    // Also init score tables for using kata go score value computation
    Board::initHash(); 
    ScoreValue::initTables();

    // Expr id, determines what games will be run and params (and if any additional params to be passed in)
    string expr_id(argv[1]);



    // 000
    // Expr to debug that all the io works
    if (expr_id == EXPR_ID_DEBUG) {
        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);   
               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);       
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.001);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_EST, //ALG_ID_KATA,            // black
            ALG_ID_KATA,             // white
            19,                  // board size
            10,                 // num games
            7.5,                // komi
            true,
            5.0,                // time per move
            32,                 // num threads 
            false,
            alg_params);   
        return 0;  
    }








    // -------------------------------------------------------------------------
    // w000 series - tuning params on dynamic programming algorithms
    // -------------------------------------------------------------------------

    //
    // w000_bts_tune_temp_no_decay
    // BTS temp with no decay
    //
    if (expr_id == EXPR_ID_W000_BTS_TEMP_CONST) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                     
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w001_bts_tune_temp_sqrt_decay
    // BTS temp with sqrt decay
    //
    if (expr_id == EXPR_ID_W001_BTS_TEMP_SQRT_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w002_bts_tune_temp_log_decay
    // BTS temp with log decay
    //
    if (expr_id == EXPR_ID_W002_BTS_TEMP_LOG_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w003_bts_tune_temp_compare
    // BTS temp (no decay vs sqrt decay vs log decay)
    //
    if (expr_id == EXPR_ID_W003_BTS_TEMP_COMPARE_DECAY) {
        int decay_type_black = stoi(argv[2]);
        int decay_type_white = stoi(argv[3]);
        string alg_id = ALG_ID_EST;

        double temp_const = 0.3; // set using w000 
        double temp_sqrt_decay = 1.0; // set using w001
        double temp_log_decay = 1.0; // set using w002 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (decay_type_black == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 0.0);  
        } else if (decay_type_black == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 1.0);  
        } else if (decay_type_black == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 2.0);  
        }

        if (decay_type_white == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 0.0);  
        } else if (decay_type_white == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 1.0);  
        } else if (decay_type_white == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 2.0);  
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_SEARCH_TEMP_DECAY_TYPE,          // hps key, black
            PARAM_SEARCH_TEMP_DECAY_TYPE_OPP);     // hps key, white 
    }

    //
    // w010_bts_tune_temp_most_visited_no_decay
    // BTS temp with no decay
    //
    if (expr_id == EXPR_ID_W010_BTS_MOST_VISITED_TEMP_CONST) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                     
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
        

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w011_bts_tune_temp_most_visited_sqrt_decay
    // BTS temp with sqrt decay
    //
    if (expr_id == EXPR_ID_W011_BTS_MOST_VISITED_TEMP_SQRT_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w012_bts_tune_temp_most_visited_log_decay
    // BTS temp with log decay
    //
    if (expr_id == EXPR_ID_W012_BTS_MOST_VISITED_TEMP_LOG) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w013_bts_tune_temp_most_visited_compare_decay
    // BTS temp (no decay vs sqrt decay vs log decay)
    //
    if (expr_id == EXPR_ID_W013_BTS_MOST_VISITED_TEMP_LOG) {
        int decay_type_black = stoi(argv[2]);
        int decay_type_white = stoi(argv[3]);
        string alg_id = ALG_ID_EST;

        double temp_const = 0.3; // set using w010 
        double temp_sqrt_decay = 1.0; // set using w011
        double temp_log_decay = 1.0; // set using w012 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        if (decay_type_black == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 0.0);  
        } else if (decay_type_black == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 1.0);  
        } else if (decay_type_black == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 2.0);  
        }

        if (decay_type_white == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 0.0);  
        } else if (decay_type_white == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 1.0);  
        } else if (decay_type_white == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 2.0);  
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_SEARCH_TEMP_DECAY_TYPE,          // hps key, black
            PARAM_SEARCH_TEMP_DECAY_TYPE_OPP);     // hps key, white 
    }

    //
    // w014_bts_tune_compare_recommendations
    // BTS compare recommendation method
    //
    if (expr_id == EXPR_ID_W014_BTS_COMPARE_RECOMMEND) {
        bool recommend_most_visited_plays_black = (stod(argv[2]) == 0.0);
        string alg_id = ALG_ID_EST;

        double temp_value = 1.0;    // set using w003
        double temp_visited = 1.0;  // set using w013

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();   
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                  
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);            
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);


        if (recommend_most_visited_plays_black) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_visited);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_value);  
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w013
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using w013
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using w013
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w003
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using w003
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using w003
        } else {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_value);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_visited);  
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w003
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using w003
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using w003
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w013
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using w013
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using w013
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            (!recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED,          // hps key, black
            (recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED_OPP);     // hps key, white 
    }

    //
    // w020_bts_tune_prior_coeff
    // w021_bts_tune_prior_coeff
    // Prior coeff
    //
    if (expr_id == EXPR_ID_W020_BTS_PRIOR_COEFF || expr_id == EXPR_ID_W021_BTS_PRIOR_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);           // set using w014
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);       // set using w014            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, coeff);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, coeff_opp);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);  

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using w014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_PRIOR_COEFF,          // hps key, black
            PARAM_PRIOR_COEFF_OPP);     // hps key, white
    }

    //
    // w030_bts_tune_eps_coeff
    // Eps coeff
    //
    if (expr_id == EXPR_ID_W030_BTS_EPS_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);           // set using w011
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);       // set using w011        
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020         
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, coeff);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, coeff_opp); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, coeff); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, coeff_opp);  

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using w014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_MENTS_ROOT_EPS,          // hps key, black
            PARAM_MENTS_ROOT_EPS_OPP);     // hps key, white
    }

    //
    // w040_ments_tune_temp
    // w041_ments_tune_temp_most_visit
    // w050_rents_tune_temp
    // w051_rents_tune_temp_most_visit
    // w060_tents_tune_temp
    // w061_tents_tune_temp_most_visit
    // Tuning temperature of DP algorithms using params from BTS which can
    //
    if (expr_id == EXPR_ID_W040_MENTS_TEMP 
        || expr_id == EXPR_ID_W041_MENTS_TEMP 
        || expr_id == EXPR_ID_W050_RENTS_TEMP
        || expr_id == EXPR_ID_W051_RENTS_TEMP
        || expr_id == EXPR_ID_W060_TENTS_TEMP
        || expr_id == EXPR_ID_W061_TENTS_TEMP) 
    {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_MENTS;
        if (expr_id == EXPR_ID_W050_RENTS_TEMP || expr_id == EXPR_ID_W051_RENTS_TEMP) {
            alg_id = ALG_ID_RENTS;
        } else if (expr_id == EXPR_ID_W060_TENTS_TEMP || expr_id == EXPR_ID_W061_TENTS_TEMP) {
            alg_id = ALG_ID_TENTS;
        } 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                    
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  

        if (expr_id == EXPR_ID_W041_MENTS_TEMP 
            || expr_id == EXPR_ID_W051_RENTS_TEMP
            || expr_id == EXPR_ID_W061_TENTS_TEMP) 
        {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // w042_ments_tune_temp_compare_recommend
    // w052_rents_tune_temp_compare_recommend
    // w062_tents_tune_temp_compare_recommend
    // Comparing most visited vs best value recommendations
    //
    if (expr_id == EXPR_ID_W042_MENTS_TEMP
        || expr_id == EXPR_ID_W052_RENTS_TEMP
        || expr_id == EXPR_ID_W062_TENTS_TEMP) 
    {
        bool recommend_most_visited_plays_black = (stod(argv[2]) == 0.0);

        string alg_id = ALG_ID_MENTS;
        if (expr_id == EXPR_ID_W052_RENTS_TEMP) {
            alg_id = ALG_ID_RENTS;
        } else if (expr_id == EXPR_ID_W062_TENTS_TEMP) {
            alg_id = ALG_ID_TENTS;
        } 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        // alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        // alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                    
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  

        if (recommend_most_visited_plays_black) {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            if (expr_id == EXPR_ID_W042_MENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);      // set using w041
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3);  // set using w040
            }
            if (expr_id == EXPR_ID_W052_RENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);      // set using w051
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);  // set using w050
            }
            if (expr_id == EXPR_ID_W062_TENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 30.0);      // set using w061
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 100.0);  // set using w060
            }
        } else {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
            if (expr_id == EXPR_ID_W042_MENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);      // set using w040
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.30);  // set using w041
            }
            if (expr_id == EXPR_ID_W052_RENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);      // set using w050
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);  // set using w051
            }
            if (expr_id == EXPR_ID_W062_TENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 100.0);      // set using w060
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 30.0);  // set using w061
            }
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            (!recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED,          // hps key, black
            (recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED_OPP);     // hps key, white 
    }

    // 
    // w070_dents_tune_value_temp
    // DENTS value temp
    // 
    if (expr_id == EXPR_ID_W070_DENTS_VALUE_TEMP) {
        double value_temp = stod(argv[2]);
        double value_temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_DENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);           // set using w014
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);       // set using w014        
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020      
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp); 

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using w014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using w014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using w014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
    }








    // -------------------------------------------------------------------------
    // x000 series - tuning params on average returns algorithms
    // -------------------------------------------------------------------------

    //
    // x000_bts_tune_temp_no_decay
    // BTS temp with no decay
    //
    if (expr_id == EXPR_ID_X000_BTS_TEMP_CONST) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                     
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x001_bts_tune_temp_sqrt_decay
    // BTS temp with sqrt decay
    //
    if (expr_id == EXPR_ID_X001_BTS_TEMP_SQRT_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x002_bts_tune_temp_log_decay
    // BTS temp with log decay
    //
    if (expr_id == EXPR_ID_X002_BTS_TEMP_LOG_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x003_bts_tune_temp_compare
    // BTS temp (no decay vs sqrt decay vs log decay)
    //
    if (expr_id == EXPR_ID_X003_BTS_TEMP_COMPARE_DECAY) {
        int decay_type_black = stoi(argv[2]);
        int decay_type_white = stoi(argv[3]);
        string alg_id = ALG_ID_EST;

        double temp_const = 0.03; // set using x000 
        double temp_sqrt_decay = 0.1; // set using x001
        double temp_log_decay = 0.1; // set using x002 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (decay_type_black == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 0.0);  
        } else if (decay_type_black == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 1.0);  
        } else if (decay_type_black == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 2.0);  
        }

        if (decay_type_white == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 0.0);  
        } else if (decay_type_white == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 1.0);  
        } else if (decay_type_white == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 2.0);  
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_SEARCH_TEMP_DECAY_TYPE,          // hps key, black
            PARAM_SEARCH_TEMP_DECAY_TYPE_OPP);     // hps key, white 
    }

    //
    // x010_bts_tune_temp_most_visited_no_decay
    // BTS temp with no decay
    //
    if (expr_id == EXPR_ID_X010_BTS_MOST_VISITED_TEMP_CONST) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                     
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
        

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x011_bts_tune_temp_most_visited_sqrt_decay
    // BTS temp with sqrt decay
    //
    if (expr_id == EXPR_ID_X011_BTS_MOST_VISITED_TEMP_SQRT_DECAY) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x012_bts_tune_temp_most_visited_log_decay
    // BTS temp with log decay
    //
    if (expr_id == EXPR_ID_X012_BTS_MOST_VISITED_TEMP_LOG) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x013_bts_tune_temp_most_visited_compare_decay
    // BTS temp (no decay vs sqrt decay vs log decay)
    //
    if (expr_id == EXPR_ID_X013_BTS_MOST_VISITED_TEMP_LOG) {
        int decay_type_black = stoi(argv[2]);
        int decay_type_white = stoi(argv[3]);
        string alg_id = ALG_ID_EST;

        double temp_const = 0.03; // set using x010 
        double temp_sqrt_decay = 0.1; // set using x011
        double temp_log_decay = 0.1; // set using x012 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                       
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);

        if (decay_type_black == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 0.0);  
        } else if (decay_type_black == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 1.0);  
        } else if (decay_type_black == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE, 2.0);  
        }

        if (decay_type_white == 0) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_const);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 0.0);  
        } else if (decay_type_white == 1) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_sqrt_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 1.0);  
        } else if (decay_type_white == 2) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_log_decay);
            alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);  
            alg_params->insert_or_assign(PARAM_SEARCH_TEMP_DECAY_TYPE_OPP, 2.0);  
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_SEARCH_TEMP_DECAY_TYPE,          // hps key, black
            PARAM_SEARCH_TEMP_DECAY_TYPE_OPP);     // hps key, white 
    }

    //
    // x014_bts_tune_compare_recommendations
    // BTS compare recommendation method
    //
    if (expr_id == EXPR_ID_X014_BTS_COMPARE_RECOMMEND) {
        bool recommend_most_visited_plays_black = (stod(argv[2]) == 0.0);
        string alg_id = ALG_ID_EST;

        double temp_value = 0.1;    // set using x003
        double temp_visited = 0.1;  // set using x013

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();   
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.25);                  
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.25);            
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);


        if (recommend_most_visited_plays_black) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_visited);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_value);  
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x013
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x013
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x013
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x003
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x003
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x003
        } else {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_value);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_visited);  
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x003
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x003
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x003
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x013
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x013
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x013
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            (!recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED,          // hps key, black
            (recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED_OPP);     // hps key, white 
    }

    //
    // x020_bts_tune_prior_coeff
    // x021_bts_tune_prior_coeff
    // Prior coeff
    //
    if (expr_id == EXPR_ID_X020_BTS_PRIOR_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);           // set using x014
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1);       // set using x014            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, coeff);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, coeff_opp);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);  

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_PRIOR_COEFF,          // hps key, black
            PARAM_PRIOR_COEFF_OPP);     // hps key, white
    }

    //
    // x030_bts_tune_eps_coeff
    // Eps coeff
    //
    if (expr_id == EXPR_ID_X030_BTS_EPS_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);           // set using x011
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1);       // set using x011        
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.00);                  // set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.00);              // set using x020         
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, coeff);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, coeff_opp); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, coeff); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, coeff_opp);  

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_MENTS_ROOT_EPS,          // hps key, black
            PARAM_MENTS_ROOT_EPS_OPP);     // hps key, white
    }

    //
    // x040_ments_tune_temp
    // x041_ments_tune_temp_most_visit
    // x050_rents_tune_temp
    // x051_rents_tune_temp_most_visit
    // x060_tents_tune_temp
    // x061_tents_tune_temp_most_visit
    // Tuning temperature of DP algorithms using params from BTS which can
    //
    if (expr_id == EXPR_ID_X040_MENTS_TEMP 
        || expr_id == EXPR_ID_X041_MENTS_TEMP 
        || expr_id == EXPR_ID_X050_RENTS_TEMP
        || expr_id == EXPR_ID_X051_RENTS_TEMP
        || expr_id == EXPR_ID_X060_TENTS_TEMP
        || expr_id == EXPR_ID_X061_TENTS_TEMP) 
    {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_MENTS;
        if (expr_id == EXPR_ID_X050_RENTS_TEMP || expr_id == EXPR_ID_X051_RENTS_TEMP) {
            alg_id = ALG_ID_RENTS;
        } else if (expr_id == EXPR_ID_X060_TENTS_TEMP || expr_id == EXPR_ID_X061_TENTS_TEMP) {
            alg_id = ALG_ID_TENTS;
        } 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                    
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.00);                  // set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.00);              // set using x020     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  

        if (expr_id == EXPR_ID_X041_MENTS_TEMP 
            || expr_id == EXPR_ID_X051_RENTS_TEMP
            || expr_id == EXPR_ID_X061_TENTS_TEMP) 
        {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // x042_ments_tune_temp_compare_recommend
    // x052_rents_tune_temp_compare_recommend
    // x062_tents_tune_temp_compare_recommend
    // Comparing most visited vs best value recommendations
    //
    if (expr_id == EXPR_ID_X042_MENTS_TEMP
        || expr_id == EXPR_ID_X052_RENTS_TEMP
        || expr_id == EXPR_ID_X062_TENTS_TEMP) 
    {
        bool recommend_most_visited_plays_black = (stod(argv[2]) == 0.0);

        string alg_id = ALG_ID_MENTS;
        if (expr_id == EXPR_ID_X052_RENTS_TEMP) {
            alg_id = ALG_ID_RENTS;
        } else if (expr_id == EXPR_ID_X062_TENTS_TEMP) {
            alg_id = ALG_ID_TENTS;
        } 

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        // alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        // alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                    
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.00);                  // set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.00);              // set using x020     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        // alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  

        if (recommend_most_visited_plays_black) {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
            if (expr_id == EXPR_ID_X042_MENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.30);      // set using x041
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.01);  // set using x040
            }
            if (expr_id == EXPR_ID_X052_RENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.30);      // set using x051
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3);  // set using x050
            }
            if (expr_id == EXPR_ID_X062_TENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);      // set using x061
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0);  // set using x060
            }
        } else {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
            if (expr_id == EXPR_ID_X042_MENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.01);      // set using x040
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.30);  // set using x041
            }
            if (expr_id == EXPR_ID_X052_RENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);      // set using x050
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.30);  // set using x051
            }
            if (expr_id == EXPR_ID_X062_TENTS_TEMP) {
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);      // set using x060
                alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0);  // set using x061
            }
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            (!recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED,          // hps key, black
            (recommend_most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED_OPP);     // hps key, white 
    }

    // 
    // x070_dents_tune_value_temp
    // DENTS value temp
    // 
    if (expr_id == EXPR_ID_X070_DENTS_VALUE_TEMP) {
        double value_temp = stod(argv[2]);
        double value_temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_DENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);           //  set using x011
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3);       //  set using x011        
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.00);                  //  set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.00);              //  set using x020      
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              //  set using x030                     
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          //  set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   //  set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               //  set using x030
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp); 

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     //  set using x014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); //  set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      //  set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      //  set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   //  set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   //  set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       //  set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       //  set using x014

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
    }








    // -------------------------------------------------------------------------
    // y000 series - testing algorithms on 9x9 go
    // -------------------------------------------------------------------------

    //
    // y000_puct_vs_kata_native
    // PUCT vs Native Kata
    //
    if (expr_id == EXPR_ID_Y000_PUCT_VS_KATA_NATIVE) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);   

        thts::run_go_games(
            expr_id,            // expr id
            algo1,            // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    //
    // y001_ar_bts_vs_kata_native
    // BTS vs Native Kata
    //
    if (expr_id == EXPR_ID_Y001_AR_BTS_VS_KATA_NATIVE) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();      
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020         
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x014

        if (algo1 == ALG_ID_KATA_NATIVE) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1);       // set using x014  
        } else {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);   
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);           // set using x014
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,            // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    //
    // y010_ar_vs_dp
    // Average returns vs Dynamic Programming
    //
    if (expr_id == EXPR_ID_Y010_AR_VS_DP) {
        double emp_plays_black = stod(argv[2]) == 0.0;
        string alg_id(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();

        if (emp_plays_black) {
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020        
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030 
            alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020                   
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030  
            alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030
        } else {
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020                
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030 
            alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020    
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030  
            alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        }


        double temp = 0.0;
        double temp_opp = 0.0;
        double value_temp = 0.0;
        double value_temp_opp = 0.0;

        if (alg_id == ALG_ID_EST && emp_plays_black) {
            temp = 0.1;         // set using x011
            temp_opp = 1.0;     // set using w011
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        }
        if (alg_id == ALG_ID_EST && !emp_plays_black) {
            temp_opp = 0.1;     // set using x011
            temp = 1.0;         // set using w011
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        }

        if (alg_id == ALG_ID_MENTS && emp_plays_black) {
            temp = 0.3;         // set using x042
            temp_opp = 1.0;     // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);     // set using w042
        }
        if (alg_id == ALG_ID_MENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x042
            temp = 1.0;         // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w042
        }

        if (alg_id == ALG_ID_RENTS && emp_plays_black) {
            temp = 0.3;         // set using x052
            temp_opp = 1.0;     // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);     // set using w052
        }
        if (alg_id == ALG_ID_RENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x052
            temp = 1.0;         // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w052
        }

        if (alg_id == ALG_ID_TENTS && emp_plays_black) {
            temp = 0.3;         // set using x062
            temp_opp = 300.0;     // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 30.0);     // set using w062
        }
        if (alg_id == ALG_ID_TENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x062
            temp = 300.0;         // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 30.0);     // set using w062
        }

        if (alg_id == ALG_ID_DENTS && emp_plays_black) {
            temp = 0.1;             // set using x011
            value_temp = 0.3;       // set using x070
            temp_opp = 1.0;         // set using w011
            value_temp_opp = 0.3;   // set using w070
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        }
        if (alg_id == ALG_ID_DENTS && !emp_plays_black) {
            temp_opp = 0.1;         // set using x011
            value_temp_opp = 0.3;   // set using x070
            temp = 1.0;             // set using w011
            value_temp = 0.3;       // set using w070
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        }

        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);  

        if (alg_id != ALG_ID_DENTS && alg_id != ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }  
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,              // black
            alg_id,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!emp_plays_black) ? "" : PARAM_USE_AVG_RETURN,
            (emp_plays_black) ? "" : PARAM_USE_AVG_RETURN_OPP);        
        return 0;
    }

    //
    // y020_ar_round_robin_with_rand
    // y030_ar_round_robin
    // Round robins (with and without random search - with random search only tests against some)
    //
    if (expr_id == EXPR_ID_Y020_AR_RR_WITH_RAND || expr_id == EXPR_ID_Y030_AR_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020        
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030
        
        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);  
        }
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0); 
        }

        if (algo1 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);     // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);     // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);     // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using x014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.3);               // set using x070                
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
        }
        if (algo2 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using x014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.3);       // set using x070
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using x014
        }

        if (algo1 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
        }
        if (algo2 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x014
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using x014
        }

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    //
    // y021_dp_round_robin_with_rand
    // y031_dp_round_robin
    // Round robins (with and without random search - with random search only tests against some)
    //
    if (expr_id == EXPR_ID_Y021_DP_RR_WITH_RAND || expr_id == EXPR_ID_Y031_DP_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.0);                  // set using w020        
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.0);              // set using w020                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030
        
        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);  
        }
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0); 
        }

        if (algo1 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 30.0);     // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 30.0); // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.3);               // set using w070                
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
        }
        if (algo2 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.3);       // set using w070
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using w014
        }

        if (algo1 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
        }
        if (algo2 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w014
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using w014
        }

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }








    // -------------------------------------------------------------------------
    // z000 series - testing algorithms on 19x19 go
    // -------------------------------------------------------------------------

    //
    // z000_puct_vs_kata_native
    // PUCT vs Native Kata
    //
    if (expr_id == EXPR_ID_Z000_PUCT_VS_KATA_NATIVE) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);   

        thts::run_go_games(
            expr_id,            // expr id
            algo1,            // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    //
    // z001_ar_bts_vs_kata_native
    // BTS vs Native Kata
    //
    if (expr_id == EXPR_ID_Z001_AR_BTS_VS_KATA_NATIVE) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();      
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020         
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
        alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
        // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);   // set using x014
        alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);   // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);       // set using x014
        // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);       // set using x014

        if (algo1 == ALG_ID_KATA_NATIVE) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1);       // set using x014  
        } else {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);   
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);           // set using x014
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,            // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    //
    // z010_ar_vs_dp
    // Average returns vs Dynamic Programming
    //
    if (expr_id == EXPR_ID_Z010_AR_VS_DP) {
        double emp_plays_black = stod(argv[2]) == 0.0;
        string alg_id(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();

        if (emp_plays_black) {
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020        
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030 
            alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.00);              // set using w020                   
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030  
            alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030
        } else {
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020                
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030 
            alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030

            alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.00);                  // set using w020    
            alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030  
            alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        }


        double temp = 0.0;
        double temp_opp = 0.0;
        double value_temp = 0.0;
        double value_temp_opp = 0.0;

        if (alg_id == ALG_ID_EST && emp_plays_black) {
            temp = 0.1;         // set using x011
            temp_opp = 1.0;     // set using w011
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        }
        if (alg_id == ALG_ID_EST && !emp_plays_black) {
            temp_opp = 0.1;     // set using x011
            temp = 1.0;         // set using w011
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        }

        if (alg_id == ALG_ID_MENTS && emp_plays_black) {
            temp = 0.3;         // set using x042
            temp_opp = 1.0;     // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);     // set using w042
        }
        if (alg_id == ALG_ID_MENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x042
            temp = 1.0;         // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w042
        }

        if (alg_id == ALG_ID_RENTS && emp_plays_black) {
            temp = 0.3;         // set using x052
            temp_opp = 1.0;     // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);     // set using w052
        }
        if (alg_id == ALG_ID_RENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x052
            temp = 1.0;         // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w052
        }

        if (alg_id == ALG_ID_TENTS && emp_plays_black) {
            temp = 0.3;         // set using x062
            temp_opp = 300.0;     // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 30.0);     // set using w062
        }
        if (alg_id == ALG_ID_TENTS && !emp_plays_black) {
            temp_opp = 0.3;     // set using x062
            temp = 300.0;         // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 30.0);     // set using w062
        }

        if (alg_id == ALG_ID_DENTS && emp_plays_black) {
            temp = 0.1;             // set using x011
            value_temp = 0.3;       // set using x070
            temp_opp = 1.0;         // set using w011
            value_temp_opp = 0.3;   // set using w070
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using w014
        }
        if (alg_id == ALG_ID_DENTS && !emp_plays_black) {
            temp_opp = 0.1;         // set using x011
            value_temp_opp = 0.3;   // set using x070
            temp = 1.0;             // set using w011
            value_temp = 0.3;       // set using w070
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0); // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0); // set using x014
        }

        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);  

        if (alg_id != ALG_ID_DENTS && alg_id != ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }  
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,              // black
            alg_id,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            5.0,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!emp_plays_black) ? "" : PARAM_USE_AVG_RETURN,
            (emp_plays_black) ? "" : PARAM_USE_AVG_RETURN_OPP);        
        return 0;
    }

    //
    // z020_ar_round_robin_with_rand
    // z030_ar_round_robin
    // Round robins (with and without random search - with random search only tests against some)
    //
    if (expr_id == EXPR_ID_Z020_AR_RR_WITH_RAND || expr_id == EXPR_ID_Z030_AR_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);                  // set using x020        
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using x030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using x030
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);              // set using x020                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using x030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using x030
        
        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);  
        }
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0); 
        }

        if (algo1 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);     // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3); // set using x042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.3);     // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.3); // set using x052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);     // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0); // set using x062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using x014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.3);               // set using x070                
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
        }
        if (algo2 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using x014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.3);       // set using x070
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using x014
        }

        if (algo1 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using x014
        }
        if (algo2 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using x014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using x014
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using x014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using x014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using x014
        }

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            5.0,               // time per move
            32,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    //
    // z021_dp_round_robin_with_rand
    // z031_dp_round_robin
    // Round robins (with and without random search - with random search only tests against some)
    //
    if (expr_id == EXPR_ID_Z021_DP_RR_WITH_RAND || expr_id == EXPR_ID_Z031_DP_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.0);                  // set using w020        
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.003);              // set using w030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.003);                   // set using w030
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.0);              // set using w020                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.003);          // set using w030 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.003);               // set using w030
        
        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);  
        }
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0); 
        }

        if (algo1 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_MENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w042
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w042
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_RENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w052
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w052
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 30.0);     // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);   
        }
        if (algo2 == ALG_ID_TENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 30.0); // set using w062
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w062
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0); 
        }

        if (algo1 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);     // set using w014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.3);               // set using w070                
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
        }
        if (algo2 == ALG_ID_DENTS) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 1.0); // set using w014
            alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.3);       // set using w070
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using w014
        }

        if (algo1 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 0.1);     // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);    // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP, 1.0);     // set using w014
        }
        if (algo2 == ALG_ID_EST) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 0.1); // set using w014
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);    // set using w014
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            // alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);      // set using w014
            alg_params->insert_or_assign(PARAM_USE_INV_SQRT_SEARCH_TEMP_OPP, 1.0);     // set using w014
            // alg_params->insert_or_assign(PARAM_USE_INV_LOG_SEARCH_TEMP_OPP, 1.0);     // set using w014
        }

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        
        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            5.0,               // time per move
            32,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }














































    // -------------------------------------------------------------------------
    // Old tests
    // -------------------------------------------------------------------------

    // 001
    // Run puct games and test out different komis on 9x9
    if (expr_id == EXPR_ID_KOMI) {
        double komi = stod(argv[2]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,             // black
            ALG_ID_KATA,             // white
            9,                  // board size
            25,                 // num games
            komi,               // komi
            true,
            15.0,               // time per move
            32,                // num threads
            false, 
            alg_params);  
        return 0;
    }

    // 008
    // Test kata returns
    if (expr_id == EXPR_ID_KATA_RECOMMEND_TEST) {
        double emp_recommender_plays_black = stod(argv[2]) == 0.0;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);    

        if (emp_recommender_plays_black) {
            alg_params->insert_or_assign(PARAM_KATA_RECOMMEND_AVG_RETURN, 1.0);
        } else { 
            alg_params->insert_or_assign(PARAM_KATA_RECOMMEND_AVG_RETURN_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,              // black
            ALG_ID_KATA,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!emp_recommender_plays_black) ? "" : PARAM_KATA_RECOMMEND_AVG_RETURN,
            (emp_recommender_plays_black) ? "" : PARAM_KATA_RECOMMEND_AVG_RETURN_OPP);        
        return 0;
    }

    // 011
    // Run puct games and test optimal number of threads
    if (expr_id == EXPR_ID_KATA_THREAD_TEST || expr_id == EXPR_ID_KATA_THREAD_TEST_WITH_DIRICHLET) {
        int threads = stoi(argv[2]);
        int threads_opp = stoi(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 20.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 20.0);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, threads);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, threads_opp);

        if (expr_id == EXPR_ID_KATA_THREAD_TEST_WITH_DIRICHLET) {
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE, 1.0);
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,             // black
            ALG_ID_KATA,             // white
            9,                  // board size
            15,                 // num games
            6.5,               // komi
            true,
            15.0,               // time per move
            32,                // num threads
            true,
            alg_params,
            NUM_THREADS_OVERRIDE,          // hps key, black
            NUM_THREADS_OVERRIDE_OPP);     // hps key, white
        return 0;
    }

    // 012
    // Run bts games and test optimal number of threads
    if (expr_id == EXPR_ID_EST_THREAD_TEST) {
        int threads = stoi(argv[2]);
        int threads_opp = stoi(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 50.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 50.0);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.01);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.01);           
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);                   
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, threads);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, threads_opp);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_EST,             // black
            ALG_ID_EST,             // white
            9,                  // board size
            15,                 // num games
            6.5,               // komi
            true,
            15.0,               // time per move
            32,                // num threads
            true,
            alg_params,
            NUM_THREADS_OVERRIDE,          // hps key, black
            NUM_THREADS_OVERRIDE_OPP);     // hps key, white
        return 0;
    }

    // 013
    // Test dirichlet noise 
    if (expr_id == EXPR_ID_DIRICHLET_NOISE) {
        double dirichlet_noise_for_black = stod(argv[2]) == 0.0;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 20.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 20.0);   
        if (dirichlet_noise_for_black) {
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE, 1.0);
        } else { 
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,              // black
            ALG_ID_KATA,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!dirichlet_noise_for_black) ? "" : PARAM_USE_DIRICHLET_NOISE,
            (dirichlet_noise_for_black) ? "" : PARAM_USE_DIRICHLET_NOISE_OPP);        
        return 0;
    }
}
