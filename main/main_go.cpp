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
static const std::string EXPR_ID_DENTS_HPS = "005_dents_hps";
static const std::string EXPR_ID_RENTS_HPS = "006_rents_hps";
static const std::string EXPR_ID_TENTS_HPS = "007_tents_hps";
static const std::string EXPR_ID_KATA_RECOMMEND_TEST = "008_test_kata_recommend";
static const std::string EXPR_ID_REC_MOST_VISITED = "009_recommend_most_visited";
static const std::string EXPR_ID_DENTS_SEARCH_TEMP_HPS = "010_dents_search_temp_hps";
static const std::string EXPR_ID_KATA_THREAD_TEST = "011_kata_thread_test";
static const std::string EXPR_ID_EST_THREAD_TEST = "012_est_thread_test";

// 100 series - final round robins on 9x9
static const std::string EXPR_ID_RAND = "100_random_9x9"; // round robin with random search incl
static const std::string EXPR_ID_RR = "101_round_robin_9x9";
static const std::string EXPR_ID_RR_W_ALIAS = "102_round_robin_w_alias_9x9";

// 200 series - round robin using params from 9x9
static const std::string EXPR_ID_19_RAND_NO_TUNE = "200_rr_with_random_19x19";
static const std::string EXPR_ID_19_RAND_NO_TUNE_W_ALIAS = "200a_rr_with_random_w_alias_19x19";
static const std::string EXPR_ID_19_RR_NO_TUNE = "201_round_robin_19x19";
static const std::string EXPR_ID_19_RR_NO_TUNE_W_ALIAS = "202_round_robin_w_alias_19x19";

// 300 series - tuning on 19x19 (focussing on algorithms that have a chance at winning)
static const std::string EXPR_ID_19_EST_HPS = "300_est_hps";
static const std::string EXPR_ID_19_DENTS_HPS = "301_dents_hps";

// 400 series - final round robins on 19x19
static const std::string EPXR_ID_19_RAND = "400_rr_with_random_19x19";
static const std::string EXPR_ID_19_RR = "401_round_robin_19x19";


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
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 50.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 50.0); 
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                 
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, 0.5);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, 0.5);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS, 1.0);
        alg_params->insert_or_assign(PARAM_USE_ALIAS_METHODS_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_RENTS, //ALG_ID_KATA,            // black
            ALG_ID_EST,             // white
            19,                  // board size
            10,                 // num games
            7.5,                // komi
            true,
            15.0,                // time per move
            64,                 // num threads
            false,
            alg_params);
        return 0;
    }

    // 001
    // Run puct games and test out different komis on 9x9
    if (expr_id == EXPR_ID_KOMI) {
        double komi = stod(argv[2]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 10.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 10.0);

        thts::run_go_games(
            expr_id,            // expr id
            ALG_ID_KATA,             // black
            ALG_ID_KATA,             // white
            9,                  // board size
            25,                 // num games
            komi,               // komi
            true,
            15.0,               // time per move
            128,                // num threads
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
            128,                 // num threads
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
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // 004
    // EST temp hps
    if (expr_id == EXPR_ID_EST_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

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
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
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
            128,                 // num threads
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
            128,                 // num threads
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
            128,                 // num threads
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
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 10.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 10.0);    

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
            128,                 // num threads
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
            128,                 // num threads
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
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
    }

    // 011
    // Run puct games and test optimal number of threads
    if (expr_id == EXPR_ID_KATA_THREAD_TEST) {
        int threads = stoi(argv[2]);
        int threads_opp = stoi(argv[3]);

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 10.0);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 10.0);
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
            15.0,               // time per move
            128,                // num threads
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
            128,                // num threads
            true,
            alg_params,
            NUM_THREADS_OVERRIDE,          // hps key, black
            NUM_THREADS_OVERRIDE_OPP);     // hps key, white
        return 0;
    }

    // 100
    // Random - testing if uniform search can outperform Katago
    if (expr_id == EXPR_ID_RAND) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
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
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.01);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.01);   
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
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

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            9,                  // board size
            25,                 // num games
            6.5,                // komi
            true,
            15.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 101
    // Round robin for 9x9
    // TODO: use tuned params
    if (expr_id == EXPR_ID_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
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
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
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

        double temp = 10.0;
        double temp_opp = 10.0;
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

        double temp = 10.0;
        double temp_opp = 10.0;
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

    // 300
    // EST temp hps
    if (expr_id == EXPR_ID_19_EST_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_EST;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.003);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.003);  
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);                  
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);   
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
            19,                  // board size
            15,                 // num games
            7.5,                // komi
            true,
            30.0,               // time per move
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_BIAS_OR_SEARCH_TEMP,          // hps key, black
            PARAM_BIAS_OR_SEARCH_TEMP_OPP);     // hps key, white
    }

    // 301
    // Dents temp hps
    if (expr_id == EXPR_ID_19_DENTS_HPS) {
        double temp = stod(argv[2]);
        double temp_opp = stod(argv[3]);

        string alg_id = ALG_ID_DENTS;

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, 500.0);         // use result from EXPR_ID_19_EST_HPS
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, 500.0);     // use result from EXPR_ID_19_EST_HPS
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
            19,                  // board size
            15,                 // num games
            7.5,                // komi
            true,
            30.0,               // time per move
            128,                 // num threads
            true,               // ments hps
            alg_params,
            PARAM_INIT_DECAY_TEMP,          // hps key, black
            PARAM_INIT_DECAY_TEMP_OPP);     // hps key, white
    }

    // 302
    // Random - testing if uniform search can outperform Katago
    // TODO: use tuned params from 300 series
    if (expr_id == EPXR_ID_19_RAND) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 500.0;
            value_temp = 20.0;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 500.0;
            value_temp_opp = 20.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp);   
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.01);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.01);   
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);  
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);             
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            25,                 // num games
            7.5,                // komi
            true,
            30.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }

    // 303
    // Round robin for 19x19
    // TODO: use tuned params from 300 series
    if (expr_id == EXPR_ID_19_RR) {
        string algo1(argv[2]);
        string algo2(argv[3]);

        double temp = 10.0;
        double temp_opp = 10.0;
        double value_temp = 20.0;
        double value_temp_opp = 20.0;
        
        if (algo1 == ALG_ID_KATA) {
            temp = 10.0;
        }
        if (algo2 == ALG_ID_KATA) {
            temp_opp = 10.0;
        }

        if (algo1 == ALG_ID_MENTS) {
            temp = 500.0;
        }
        if (algo2 == ALG_ID_MENTS) {
            temp_opp = 500.0;
        }

        if (algo1 == ALG_ID_RENTS) {
            temp = 500.0;
        }
        if (algo2 == ALG_ID_RENTS) {
            temp_opp = 500.0;
        }

        if (algo1 == ALG_ID_TENTS) {
            temp = 500.0;
        }
        if (algo2 == ALG_ID_TENTS) {
            temp_opp = 500.0;
        }

        if (algo1 == ALG_ID_DENTS) {
            temp = 500.0;
            value_temp = 20.0;
        }
        if (algo2 == ALG_ID_DENTS) {
            temp_opp = 500.0;
            value_temp_opp = 20.0;
        }

        if (algo1 == ALG_ID_EST) {
            temp = 500.0;
        }
        if (algo2 == ALG_ID_EST) {
            temp_opp = 500.0;
        }

        shared_ptr<thts::GoAlgParams> alg_params = make_shared<thts::GoAlgParams>();
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP, temp);
        alg_params->insert_or_assign(PARAM_BIAS_OR_SEARCH_TEMP_OPP, temp_opp); 
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE, 0.01);      
        alg_params->insert_or_assign(PARAM_DECAY_TEMP_ROOT_NODE_VISITS_SCALE_OPP, 0.01);   
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE, 0.05);      
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_VISITS_SCALE_OPP, 0.05);               
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF, 0.5);            
        alg_params->insert_or_assign(PARAM_PRIOR_COEFF_OPP, 0.5);       
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID, 1.0);
        // alg_params->insert_or_assign(PARAM_DECAY_TEMP_USE_SIGMOID_OPP, 1.0);   
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP, value_temp);                
        alg_params->insert_or_assign(PARAM_INIT_DECAY_TEMP_OPP, value_temp_opp);                 
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS, 1.0);                                    
        alg_params->insert_or_assign(PARAM_MENTS_ROOT_EPS_OPP, 1.0); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS, 0.03); 
        alg_params->insert_or_assign(PARAM_MENTS_EPS_OPP, 0.03);   
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN, 1.0);
        alg_params->insert_or_assign(PARAM_USE_AVG_RETURN_OPP, 1.0);

        thts::run_go_games(
            expr_id,            // expr id
            algo1,              // black
            algo2,              // white
            19,                  // board size
            50,                 // num games
            7.5,                // komi
            true,
            30.0,               // time per move
            128,                 // num threads
            false,              // NOT running ments hps
            alg_params);        
        return 0;
    }
}
