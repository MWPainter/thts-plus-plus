#include "go/run_go.h"

#include "KataGo/cpp/game/board.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// 000 series - tuning on 9x9 go
static const std::string EXPR_ID_DEBUG = "000_debug";
static const std::string EXPR_ID_KOMI = "001_komi_9x9";
static const std::string EXPR_ID_AVG_RETURN = "002_avg_return_test";
static const std::string EXPR_ID_MENTS_HPS = "003_ments_hps";
static const std::string EXPR_ID_EST_HPS = "004_est_hps";
static const std::string EXPR_ID_EST_PRIOR_HPS = "004a_est_prior_hps";
static const std::string EXPR_ID_EST_EPS_HPS = "004b_est_eps_hps";
static const std::string EXPR_ID_DENTS_HPS = "005_dents_hps";
static const std::string EXPR_ID_RENTS_HPS = "006_rents_hps";
static const std::string EXPR_ID_TENTS_HPS = "007_tents_hps";
static const std::string EXPR_ID_KATA_RECOMMEND_TEST = "008_test_kata_recommend";
static const std::string EXPR_ID_REC_MOST_VISITED = "009_recommend_most_visited";
static const std::string EXPR_ID_DENTS_SEARCH_TEMP_HPS = "010_dents_search_temp_hps";
static const std::string EXPR_ID_KATA_THREAD_TEST = "011_kata_thread_test";
static const std::string EXPR_ID_KATA_THREAD_TEST_WITH_DIRICHLET = "011a_kata_thread_test_with_dirichlet";
static const std::string EXPR_ID_EST_THREAD_TEST = "012_est_thread_test";
static const std::string EXPR_ID_DIRICHLET_NOISE = "013_dirichlet_noise";
static const std::string EXPR_ID_PUCT_BIAS_HPS = "014_puct_bias_hps";
static const std::string EXPR_ID_KATA_VS_NATIVE = "015_compare_with_native";
static const std::string EXPR_ID_KATA_VS_NATIVE_B = "015b_compare_with_native";
static const std::string EXPR_ID_KATA_VS_NATIVE_19 = "016_compare_with_native_19";
static const std::string EXPR_ID_KATA_VS_NATIVE_19_B = "016b_compare_with_native_19";

// 100 series - final round robins on 9x9
static const std::string EXPR_ID_RAND = "100_random_9x9"; // round robin with random search incl
static const std::string EXPR_ID_RR = "101_round_robin_9x9";
static const std::string EXPR_ID_RR_W_DIRICHLET = "101a_round_robin_with_dirichlet_9x9";
static const std::string EXPR_ID_RR_W_ALIAS = "102_round_robin_w_alias_9x9";

// 200 series - round robin using params from 9x9
static const std::string EXPR_ID_19_RAND_NO_TUNE = "200_rr_with_random_19x19";
static const std::string EXPR_ID_19_RAND_NO_TUNE_W_ALIAS = "200a_rr_with_random_w_alias_19x19";
static const std::string EXPR_ID_19_RR_NO_TUNE = "201_round_robin_19x19";
static const std::string EXPR_ID_19_RR_NO_TUNE_W_ALIAS = "202_round_robin_w_alias_19x19";

// x00 series - final runs, tuning
static const std::string EXPR_ID_BTS_TEMP_CONST = "x00_bts_tune_temp_no_decay";
static const std::string EXPR_ID_BTS_TEMP_CONST_LOWER_PRIOR = "x00a_bts_tune_temp_no_decay";
static const std::string EXPR_ID_BTS_TEMP_DECAY = "x01_bts_tune_temp_sqrt_decay";
static const std::string EXPR_ID_BTS_TEMP_DECAY_LOWER_PRIOR = "x01a_bts_tune_temp_sqrt_decay";
static const std::string EXPR_ID_BTS_TEMP_VERSUS = "x02_bts_tune_temp_compare";
static const std::string EXPR_ID_BTS_PRIOR_COEFF = "x03_bts_tune_prior_coeff";
static const std::string EXPR_ID_BTS_EPS_COEFF = "x04_bts_tune_eps_coeff";
static const std::string EXPR_ID_BTS_ROOT_EPS_COEFF = "x05_bts_tune_root_eps_coeff";
// bts num threads
// retry dp+most_visited vs dp+largetst_val vs ar+most_visited vs ar+largest_val
static const std::string EXPR_ID_PUCT_NUM_THREADS = "x08_puct_tune_threads";
// puct dirichlet noise
static const std::string EXPR_ID_MENTS_TEMP = "x12_ments_tune_temp";
static const std::string EXPR_ID_RENTS_TEMP = "x13_rents_tune_temp";
static const std::string EXPR_ID_TENTS_TEMP = "x14_tents_tune_temp";
// dents value temp (sqrt decay?)

// y00 series - final runs, round robins on 9x9
// +rand rr
// no alias
static const std::string EXPR_ID_ROUND_ROBIN_9_ALIAS = "y02_round_robin_with_alias";
// alias
// compare with native (9x9) - puct/bts w/time + puct/bts w/1600 trials

// z00 series - final runs, round robins on 19x19
// +rand rr
// no alias
// alias
// puct compare with native (19x19) - puct/bts w/time + puct/bts w/1600 trials



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
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);                
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.01);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.01); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.01); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.01);   
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.5);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.5);  
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);

        // alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 128);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_EST, //ALG_ID_KATA,            // black
            ALG_ID_KATA,             // white
            9,                  // board size
            10,                 // num games
            6.5,                // komi
            true,
            2.5,                // time per move
            32,                 // num threads 
            false,
            alg_params);   
        return 0;  
    }

    // -------------------------------------------------------------------------
    // x00 series - tuning params
    // -------------------------------------------------------------------------

    //
    // x00 - bts_tune_temp_no_decay
    // BTS temp
    //
    if (expr_id == EXPR_ID_BTS_TEMP_CONST || expr_id == EXPR_ID_BTS_TEMP_CONST_LOWER_PRIOR) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;
        double prior_coeff = 1.0;
        if (expr_id == EXPR_ID_BTS_TEMP_CONST_LOWER_PRIOR) {
            prior_coeff = 0.2;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                     
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, prior_coeff);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, prior_coeff);          
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x01 - bts_tune_temp_sqrt_decay
    // BTS temp
    //
    if (expr_id == EXPR_ID_BTS_TEMP_DECAY || expr_id == EXPR_ID_BTS_TEMP_DECAY_LOWER_PRIOR) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;
        double prior_coeff = 1.0;
        if (expr_id == EXPR_ID_BTS_TEMP_DECAY_LOWER_PRIOR) {
            prior_coeff = 0.2;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);             
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, prior_coeff);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, prior_coeff);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x02 - bts_tune_temp_compare
    // BTS temp (no decay vs sqrt decay)
    //
    if (expr_id == EXPR_ID_BTS_TEMP_VERSUS) {
        bool const_plays_black = (stod(argv[2]) == 0.0);
        string alg_id = ALG_ID_EST;

        double temp_const = 0.3;
        double temp_decay = 3.0;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);               
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (const_plays_black) {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_const);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_decay);  
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
        } else {
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp_decay);
            alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_const);  
            alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            (!const_plays_black) ? "" : PARAM_USE_CONST_SEARCH_TEMP,          // hps key, black
            (const_plays_black) ? "" : PARAM_USE_CONST_SEARCH_TEMP_OPP);     // hps key, white 
    }

    //
    // x03 - bts_tune_prior_coeff
    // Prior coeff
    //
    if (expr_id == EXPR_ID_BTS_PRIOR_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0);              
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

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x04 - bts_tune_eps_coeff
    // Eps coeff
    //
    if (expr_id == EXPR_ID_BTS_EPS_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0);              
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.75);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.75);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, coeff);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, coeff_opp); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, coeff); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, coeff_opp);  

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x05 - bts_tune_root_eps_coeff
    // Eps coeff
    //
    if (expr_id == EXPR_ID_BTS_ROOT_EPS_COEFF) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 3.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 3.0);              
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.75);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.75);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, coeff);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, coeff_opp); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   
        
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x06
    // 
    //  

    //
    // x07
    // 
    //  

    //
    // x08 - puct_tune_threads
    // Puct num threads
    //  
    if (expr_id == EXPR_ID_PUCT_NUM_THREADS) {
        int threads = stoi(argv[2]);
        int threads_opp = stoi(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 110.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 110.0);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, threads);
        alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, threads_opp);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,             // black
            ALG_ID_KATA,             // white
            9,                  // board size
            15,                 // num games
            6.5,               // komi
            true,
            2.5,               // time per move
            32,                // num threads
            true,
            alg_params,
            NUM_THREADS_OVERRIDE,          // hps key, black
            NUM_THREADS_OVERRIDE_OPP);     // hps key, white
        return 0;
    }

    //
    // x09
    // 
    //  

    //
    // x10
    // 
    //  

    //
    // x11
    // 
    //  

    //
    // x12 - ments_tune_temp
    // MENTS temp
    //
    if (expr_id == EXPR_ID_MENTS_TEMP) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_MENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);             
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);      
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.3);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.3); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x13 - rents_tune_temp
    // RENTS temp
    //
    if (expr_id == EXPR_ID_RENTS_TEMP) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_RENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);             
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);       
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.3);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.3); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
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
    // x14 - tents_tune_temp
    // TENTS temp
    //
    if (expr_id == EXPR_ID_TENTS_TEMP) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_TENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);             
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 1.0);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 1.0);        
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.3);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.3); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            2.5,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // -------------------------------------------------------------------------
    // y00 series - 9x9 round robins
    // -------------------------------------------------------------------------

    //
    // y00
    // Round robin with random search
    //

    //
    // y01
    // Round robin 9x9
    //

    //
    // y02 - round_robin_with_alias
    // Round robin 9x9 with alias
    //
    if (expr_id == EXPR_ID_ROUND_ROBIN_9_ALIAS) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 110.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 110.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 0.3;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 0.3;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 0.3;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 0.3;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 3.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 3.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 0.3;
            value_temp = 0.3;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 0.3;
            value_temp_opp = 0.3;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 0.3;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 0.3;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.75);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.75);       
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.3);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.3); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.001); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.001);   

        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);       
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp); 

        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP, 1.0);  
        alg_params->insert_or_assign(PARAM_USE_CONST_SEARCH_TEMP_OPP, 1.0);  
        
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
    // z00 series - 19x19 round robins
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // 000 series - old tests - tuning params
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

    // 002
    // Test empirical 
    if (expr_id == EXPR_ID_AVG_RETURN) {
        double emp_plays_black = stod(argv[2]) == 0.0;
        string alg_id(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 20.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 20.0);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 20.0);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 20.0);              
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   

        if (emp_plays_black) {
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        } else { 
            alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,              // black
            alg_id,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!emp_plays_black) ? "" : PARAM_USE_AVG_RETURN,
            (emp_plays_black) ? "" : PARAM_USE_AVG_RETURN_OPP);        
        return 0;
    }

    // 003
    // Ments temp hps
    if (expr_id == EXPR_ID_MENTS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_MENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 20.0);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 20.0);            
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    //
    // 004
    // EST temp hps
    if (expr_id == EXPR_ID_EST_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);              
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            5.0,               // time per move
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // 
    // 004a
    // EST temp hps
    if (expr_id == EXPR_ID_EST_PRIOR_HPS) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 10.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 10.0);              
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, coeff);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, coeff_opp);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.03);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            5.0,               // time per move
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_PRIOR_COEFF,          // hps key, black
            PARAM_PRIOR_COEFF_OPP);     // hps key, white
    }

    // 
    // 004b
    // EST temp hps
    if (expr_id == EXPR_ID_EST_EPS_HPS) {
        double coeff = stod(argv[2]);
        double coeff_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 10.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 10.0);              
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.5);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, coeff);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, coeff_opp); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, coeff); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, coeff_opp);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            5.0,               // time per move
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_MENTS_ROOT_EPS,          // hps key, black
            PARAM_MENTS_ROOT_EPS_OPP);     // hps key, white
    }

    // 005
    // Dents temp hps
    if (expr_id == EXPR_ID_DENTS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_DENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 50.0);         // use result from EXPR_ID_EST_HPS
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 50.0);     // use result from EXPR_ID_EST_HPS
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, temp_opp);             
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);    
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
    }

    // 006
    // Rents temp hps
    if (expr_id == EXPR_ID_RENTS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_RENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 20.0);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 20.0);              
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // 007
    // Rents temp hps
    if (expr_id == EXPR_ID_TENTS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_TENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 20.0);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 20.0);              
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
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

    // 009
    // Test most visited for energy search algorithms
    if (expr_id == EXPR_ID_REC_MOST_VISITED) {
        double most_visited_plays_black = stod(argv[2]) == 0.0;
        string alg_id(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 50.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 50.0);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 1.0);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 1.0);              
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        if (most_visited_plays_black) {
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED, 1.0);
        } else { 
            alg_params->insert_or_assign(PARAM_RECOMMEND_MOST_VISITED_OPP, 1.0);
        }

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,              // black
            alg_id,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,     // running "hps" -> i.e. two runs would have same folder names => folder names need to use params
            alg_params,
            (!most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED,
            (most_visited_plays_black) ? "" : PARAM_RECOMMEND_MOST_VISITED_OPP);        
        return 0;
    }

    // 010
    // Dents search temp hps
    if (expr_id == EXPR_ID_DENTS_SEARCH_TEMP_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_DENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);         // use result from EXPR_ID_EST_HPS
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);     // use result from EXPR_ID_EST_HPS
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.1);                  // use result from EXP_ID_DENTS_HPS
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.1);          // use result from EXP_ID_DENTS_HPS            
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);    
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
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

    // 014
    // Puct bias hps
    if (expr_id == EXPR_ID_PUCT_BIAS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_KATA;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            alg_id,            // black
            alg_id,              // white
            9,                  // board size
            15,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // 015
    // Puct vs native kata
    if (expr_id == EXPR_ID_KATA_VS_NATIVE || expr_id == EXPR_ID_KATA_VS_NATIVE_B) {
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
            25,                 // num games
            6.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    // 016
    // Puct vs native kata (on 19x19)
    if (expr_id == EXPR_ID_KATA_VS_NATIVE_19 || expr_id == EXPR_ID_KATA_VS_NATIVE_19_B) {
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
            25,                 // num games
            7.5,                // komi
            false,              // limit by number of moves, not time
            1600,               // num moves allowed
            32,                 // num threads
            false,               // ments hps
            alg_params);
    }

    // -------------------------------------------------------------------------
    // 100 series - old tests - round robins
    // -------------------------------------------------------------------------

    // 100
    // Random - testing if uniform search can outperform Katago
    if (expr_id == EXPR_ID_RAND) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 110.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 110.0;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 10.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);                
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 2.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 2.5);                  
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 0.01);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 0.01); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.01); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.01);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        } 
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            5.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 101
    // Round robin for 9x9
    // TODO: use tuned params
    if (expr_id == EXPR_ID_RR || expr_id == EXPR_ID_RR_W_DIRICHLET) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 110.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 110.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 0.5;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 50.0;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 50.0;
            value_temp = 0.5;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 50.0;
            value_temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 50.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                      
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        if (expr_id == EXPR_ID_RR_W_DIRICHLET) {
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE, 1.0);
            alg_params->insert_or_assign(PARAM_USE_DIRICHLET_NOISE_OPP, 1.0);
        }

        //if (algo1 == ALG_ID_KATA) {
        //    alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        //} 
        //if (algo2 == ALG_ID_KATA) {
        //    alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        //}

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            32,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 102
    // Round robin for 9x9, with alias method
    // TODO: use tuned params
    if (expr_id == EXPR_ID_RR_W_ALIAS) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 110.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 110.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 0.5;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 50.0;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 50.0;
            value_temp = 0.5;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 50.0;
            value_temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 50.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                      
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);                
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        } 
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            50,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // -------------------------------------------------------------------------
    // 200 series - old tests - round robins on 19x19
    // -------------------------------------------------------------------------

    // 200
    // Random - testing if uniform search can outperform Katago
    if (expr_id == EXPR_ID_19_RAND_NO_TUNE || expr_id == EXPR_ID_19_RAND_NO_TUNE_W_ALIAS) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 110.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 110.0;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 50.0;
            value_temp = 20.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 50.0;
            value_temp_opp = 20.0;
        }
        
        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);             
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 5.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 5.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        if (expr_id == EXPR_ID_19_RAND_NO_TUNE_W_ALIAS) {
            alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
            alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);
        }

        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        } 
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            25,                 // num games
            7.5,                // komi
            true,
            15.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 201
    // Round robin for 19x19
    // TODO: re-use tuned params from 101
    if (expr_id == EXPR_ID_19_RR_NO_TUNE) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 110.0;
        double temp_opp = 110.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 0.5;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 50.0;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 50.0;
            value_temp = 0.5;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 50.0;
            value_temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 50.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                 
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        } 
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            15.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 202
    // Round robin for 19x19
    // TODO: re-use tuned params from 101
    if (expr_id == EXPR_ID_19_RR_NO_TUNE_W_ALIAS) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 110.0;
        double temp_opp = 110.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 0.5;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 50.0;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 50.0;
            value_temp = 0.5;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 50.0;
            value_temp_opp = 0.5;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 50.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 50.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                 
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        if (algo1 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE, 32);
        } 
        if (algo2 == ALG_ID_KATA) {
            alg_params->insert_or_assign(NUM_THREADS_OVERRIDE_OPP, 32);
        }

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            15.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }
}
