#include "toy_envs/run_id.h"

#include "toy_envs/d_chain_env.h"
#include "toy_envs/frozen_lake_env.h"
#include "toy_envs/sailing_env.h"

#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/puct_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "algorithms/uct/uct_logger.h"
#include "algorithms/ments/ments_logger.h"
#include "algorithms/ments/dbments_logger.h"

#include "thts_env.h"

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/uct/hmcts_decision_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
#include "algorithms/est/est_decision_node.h"

#include <stdexcept>

using namespace std;

namespace thts {
    /**
     * Default constructor
    */
    RunID::RunID() {}

    /**
     * Initialised constructor
    */
    RunID::RunID(
        string env_id,
        string env_instance_id,
        string expr_id,
        string alg_id,
        unordered_map<string, double> alg_params,
        int num_trials,
        int max_trial_length,
        int trials_log_delta,
        int mc_eval_trials_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads) :
            env_id(env_id),
            env_instance_id(env_instance_id),
            expr_id(expr_id),
            alg_id(alg_id),
            alg_params(alg_params),
            num_trials(num_trials),
            max_trial_length(max_trial_length),
            trials_log_delta(trials_log_delta),
            mc_eval_trials_delta(mc_eval_trials_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads)
    {
    }

    /**
     * Create and return the env
    */
    shared_ptr<ThtsEnv> RunID::get_env() {
        if (env_id == DCHAIN_ENV_ID) {
            if (env_instance_id == D_10_ID) {
                return make_shared<DChainEnv>(10, 1.0);
            } else if (env_instance_id == D_10_FF_ID) {
                return make_shared<DChainEnv>(10, 0.8);
            } else if (env_instance_id == D_10_HALF_ID) {
                return make_shared<DChainEnv>(10, 0.5);
            } else if (env_instance_id == D_20_ID) {
                return make_shared<DChainEnv>(20, 1.0);
            } else if (env_instance_id == D_20_FF_ID) {
                return make_shared<DChainEnv>(20, 0.8);
            } else if (env_instance_id == D_20_HALF_ID) {
                return make_shared<DChainEnv>(20, 0.5);
            } else {
                throw runtime_error("Invalid DChain instance id.");
            }
        }

        if (env_id == FL_ENV_ID) {
            if (env_instance_id == FL_8x8) {
                return make_shared<FrozenLakeEnv>(8, 8, FL_RAND_8X8_MAP, 0.99);
            } else if (env_instance_id == FL_8x8_TEST) {
                return make_shared<FrozenLakeEnv>(8, 8, FL_RAND_8X8_TEST_MAP, 0.99);
            } else if (env_instance_id == FL_8x12) { 
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X12_MAP, 0.99);
            } else if (env_instance_id == FL_8x12_TEST) {
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X12_TEST_MAP, 0.99);
            } else if (env_instance_id == FL_8x16_TEST) {
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X16_TEST_MAP, 0.99);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        if (env_id == SAILING_ENV_ID) {
            if (env_instance_id == S_6_ID) {
                return make_shared<SailingEnv>(6, 6);
            } else if (env_instance_id == S_6_TEST_ID) {
                return make_shared<SailingEnv>(6, 6, SE);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        throw runtime_error("Error in RunID get_env");
    }

    /**
     * Create thts manager
    */
    shared_ptr<ThtsManager> RunID::get_thts_manager(shared_ptr<ThtsEnv> env) {
        if (alg_id == ALG_ID_UCT) {
            UctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.bias = alg_params.at(PARAMS_ID_UCT_BIAS);
            return make_shared<UctManager>(manager_args);
        } 
        if (alg_id == ALG_ID_PUCT) {
            PuctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.bias = alg_params.at(PARAMS_ID_UCT_BIAS);
            return make_shared<PuctManager>(manager_args);
        }
        if (alg_id == ALG_ID_HMCTS) {
            HmctsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.bias = alg_params.at(PARAMS_ID_UCT_BIAS);
            manager_args.total_budget = alg_params.at(PARAMS_ID_HMCTS_BUDGET);
            manager_args.uct_budget_threshold = alg_params.at(PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD);
            manager_args.recommend_most_visited = false;
            return make_shared<HmctsManager>(manager_args);
        }
        if (alg_id == ALG_ID_MENTS || 
            alg_id == ALG_ID_RENTS || 
            alg_id == ALG_ID_TENTS) 
        {
            MentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            return make_shared<MentsManager>(manager_args);
        }
        if (alg_id == ALG_ID_DENTS ||
            alg_id == ALG_ID_EST) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.value_temp_init = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            if (alg_params.find(PARAMS_ID_DENTS_IGNORE_DECAY) != alg_params.end()) {
                manager_args.value_temp_decay_fn = decayed_temp_no_decay;
            }
            if (alg_params.find(PARAMS_ID_DENTS_TEMP) != alg_params.end()) {
                manager_args.value_temp_init = alg_params.at(PARAMS_ID_DENTS_TEMP);
            }
            return make_shared<DentsManager>(manager_args);
        }
        if (alg_id == ALG_ID_DBMENTS) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.value_temp_init = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            manager_args.value_temp_decay_fn = decayed_temp_no_decay;
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            return make_shared<DentsManager>(manager_args);
        }

        throw runtime_error("Error in RunID get_thts_manager");
    }

    /**
     * Return a root search node
    */
    shared_ptr<ThtsDNode> RunID::get_root_search_node(shared_ptr<ThtsEnv> env, shared_ptr<ThtsManager> manager) {
        if (alg_id == ALG_ID_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_PUCT) {
            shared_ptr<PuctManager> puct_manager = static_pointer_cast<PuctManager>(manager);
            return make_shared<PuctDNode>(puct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_HMCTS) {
            shared_ptr<HmctsManager> hmcts_manager = static_pointer_cast<HmctsManager>(manager);
            return make_shared<HmctsDNode>(hmcts_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_MENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<MentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_RENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<RentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_TENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<TentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
            shared_ptr<DentsManager> dents_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<DentsDNode>(dents_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_EST) {
            shared_ptr<DentsManager> ments_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<EstDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }

        throw runtime_error("Error in RunID get_root_search_node");
    }

    /**
     * Returns a logger to use with this run
    */
    shared_ptr<ThtsLogger> RunID::get_logger() {
        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_PUCT || alg_id == ALG_ID_HMCTS) {
            shared_ptr<ThtsLogger> logger = make_shared<UctLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        } 
        if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS || alg_id == ALG_ID_RENTS || alg_id == ALG_ID_TENTS) {
            shared_ptr<ThtsLogger> logger = make_shared<MentsLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        }
        if (alg_id == ALG_ID_EST) {
            shared_ptr<ThtsLogger> logger = make_shared<DBMentsLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        }

        throw runtime_error("Error in RunID get_logger");
    }


    /**
     * Gets a list of RunID objects from a given expr id
    */
    shared_ptr<vector<RunID>> get_run_ids_from_expr_id(string expr_id) {
        shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

        // debug expr id for debugging
        if (expr_id == DEBUG_EXPR_ID) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 5;
            int num_threads = 16;
            int eval_threads = 32;

            double default_q_value = -20.0;

            vector<string> alg_ids = { ALG_ID_RENTS };
            vector<double> temps = { 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    unordered_map<string,double> alg_params = {
                            {PARAMS_ID_MENTS_TEMP, temp}, 
                            {PARAMS_ID_MENTS_EPSILON, 0.1}, 
                            {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                        };
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            return run_ids;
        }


        // expr_id: D001_LEN10
        // Runs algorithms with varying parameters on the 10-chain
        if (expr_id == D001_LEN10) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_10_ID, D_10_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {

                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }


                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_RENTS};
                temps = { 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_TENTS};
                temps = { 10.0, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
                temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_HMCTS};
                vector<int> uct_thresholds = { 1, 3, 10, 30, 100, 300, 1000};
                for (string alg_id : alg_ids) {
                    for (int thresh : uct_thresholds) {
                        unordered_map<string,double> alg_params = 
                        {
                            {PARAMS_ID_UCT_BIAS, 100.0},
                            {PARAMS_ID_HMCTS_BUDGET, num_trials},
                            {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, thresh},
                        };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

            }

            return run_ids;
        }


        // expr_id: D003_LEN20
        // Runs algorithms with varying parameters on the 20-chain
        if (expr_id == D003_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_20_ID, D_20_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {
                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }


                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_RENTS};
                temps = { 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_TENTS};
                temps = { 10.0, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
                temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D005_LEN20
        // Runs algorithms with varying parameters on the 20-chain
        if (expr_id == D005_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_20_ID, D_20_HALF_ID};
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {
                
                vector<string> alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            string env_instance_id = D_20_ID;

            vector<string> alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
            vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
            vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D006_LEN20
        // 20-chain with DENTS without decaying temp
        if (expr_id == D006_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            string env_instance_id = D_20_ID;
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_DENTS};
            vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
            vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_DENTS_IGNORE_DECAY, 1.0},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D006_LEN20
        // 20-chain with DENTS tuned
        if (expr_id == D007_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            string env_instance_id = D_20_ID;
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            string alg_id = ALG_ID_DENTS;
            unordered_map<string,double> alg_params = 
                {
                    {PARAMS_ID_MENTS_TEMP, 0.5}, 
                    {PARAMS_ID_MENTS_EPSILON, 0.01},
                    {PARAMS_ID_DENTS_TEMP, 10.0},
                };
            run_ids->push_back(RunID(
                env_id,
                env_instance_id,
                expr_id,
                alg_id,
                alg_params,
                num_trials,
                max_trial_length,
                trials_log_delta,
                mc_eval_trials_delta,
                rollouts_per_mc_eval,
                num_repeats,
                num_threads,
                eval_threads));
                
            return run_ids;
        }

        // expr id: D021_LEN10_PAPER = "100_len_10_main_paper"
        // rerunning with specific parameters with more replicates to make curves smoother for nice plots
        if (expr_id == D021_LEN10_PAPER) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_10_ID, D_10_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 100;
            int num_threads = 16;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {

                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                alg_ids = {ALG_ID_DENTS};
                temps = { 1.0 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                // adding DENTS with fixed beta(m)=alpha to demonstrate dents mimicing ments 
                // (and just copied MENTS params without tunring)
                alg_ids = {ALG_ID_DBMENTS};
                temps = { 1.0 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                // HMCTS
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_UCT_BIAS, 100.0},
                        {PARAMS_ID_HMCTS_BUDGET, num_trials},
                        {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, 30},
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    ALG_ID_HMCTS,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }










        // expr id: FL12_051_HPS
        // Runs a hyperparameter search on all algos for frozen lake
        if (expr_id == FL12_051_HPS) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (string alg_id : alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS };
            vector<double> temps = {100.0, 10.0, 1.0, 0.1, 0.01, 0.001};
            vector<double> epss = {2.0, 1.0, 0.3, 0.1, 0.03, 0.01};
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));       
                    }
                }
            }

            vector<int> uct_thresholds = { 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
            uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (double bias : uct_biases) {
                for (int thresh : uct_thresholds) {
                    unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_UCT_BIAS, bias},
                        {PARAMS_ID_HMCTS_BUDGET, num_trials},
                        {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, thresh},
                    };
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        ALG_ID_HMCTS,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            return run_ids;
        }

        // expr id: FL12_051A_HPS
        // Tunes the dents temperature after the rest of the params are set
        if (expr_id == FL12_051A_HPS) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            string alg_id = ALG_ID_DENTS;
            double m_temp = 0.1;
            double eps = 1.0;
            vector<double> d_temps = {100.0, 10.0, 1.0, 0.1};

            for (double d_temp : d_temps) {
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, m_temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps},
                        {PARAMS_ID_DENTS_TEMP, d_temp}, 
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));    
            }

            return run_ids;
        }

        // expr id: FL12_052_TEST
        // Test envs for hps selected params
        if (expr_id == FL12_052_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 500000;
            int max_trial_length = 100; 
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 0.001;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 0.1;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in FL12_052_TEST");
                }
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps}
                    };
                if (alg_id == ALG_ID_DENTS) {
                    alg_params.insert_or_assign(PARAMS_ID_DENTS_TEMP, 1.0);
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            // adding DENTS with fixed beta(m)=alpha to demonstrate dents mimicing ments 
            // (and just copied MENTS params without tunring)
            alg_ids = {ALG_ID_DBMENTS};
            vector<double> temps = { 0.01 };
            vector<double> epss = { 1.0 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            unordered_map<string,double> alg_params = 
            {
                {PARAMS_ID_UCT_BIAS, UctManagerArgs::USE_AUTO_BIAS},
                {PARAMS_ID_HMCTS_BUDGET, num_trials},
                {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, 3000},
            };
            run_ids->push_back(RunID(
                env_id,
                env_instance_id,
                expr_id,
                ALG_ID_HMCTS,
                alg_params,
                num_trials,
                max_trial_length,
                trials_log_delta,
                mc_eval_trials_delta,
                rollouts_per_mc_eval,
                num_repeats,
                num_threads,
                eval_threads));

            return run_ids;
        }

        // expr id: FL16_050_TEST
        // Test envs for hps selected params
        if (expr_id == FL16_050_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x16_TEST;
            int num_trials = 500000;
            int max_trial_length = 100; 
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 0.001;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 0.1;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in FL12_052_TEST");
                }
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps}
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: FL8_05X_SENS
        // Experiments showing sensitivity of ments temperature
        // Runs a hyperparameter search on all algos for frozen lake
        if (expr_id == FL8_053_SENS ||
            expr_id == FL8_054_SENS ||
            expr_id == FL8_055_SENS ||
            expr_id == FL8_056_SENS ||
            expr_id == FL8_057_SENS ||
            expr_id == FL8_058_SENS ||
            expr_id == FL8_059_SENS)
        {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x8;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 15;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_MENTS, ALG_ID_EST, ALG_ID_DENTS, ALG_ID_RENTS, ALG_ID_TENTS};
            alg_ids = {ALG_ID_RENTS, ALG_ID_TENTS};
            double temp = 1.0;
            double eps = 1.0;
            if (expr_id == FL8_053_SENS) {
                temp = 1.0;
            } else if (expr_id == FL8_054_SENS) {
                temp = 0.5;
            } else if (expr_id == FL8_055_SENS) {
                temp = 0.1;
            } else if (expr_id == FL8_056_SENS) {
                temp = 0.05;
            } else if (expr_id == FL8_057_SENS) {
                temp = 0.01;
            } else if (expr_id == FL8_058_SENS) {
                temp = 0.005;
            } else if (expr_id == FL8_059_SENS) {
                temp = 0.001;
            } else {
                throw runtime_error("something happened");
            }
            
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps},
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));  
            }

            return run_ids;
        }










        

        // expr id: S6_091_HPS
        // Runs a hyperparameter search on all algos for sailing 6x6 env
        if (expr_id == S6_091_HPS) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 250000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            double default_q_value = -200.0;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (string alg_id : alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS };
            vector<double> temps = {100.0, 10.0, 1.0, 0.1, 0.01, 0.001};
            vector<double> epss = {2.0, 1.0, 0.3, 0.1}; //, 0.03, 0.01};
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));       
                    }
                }
            }

            vector<int> uct_thresholds = { 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
            uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (double bias : uct_biases) {
                for (int thresh : uct_thresholds) {
                    unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_UCT_BIAS, bias},
                        {PARAMS_ID_HMCTS_BUDGET, num_trials},
                        {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, thresh},
                        {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                    };
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        ALG_ID_HMCTS,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            return run_ids;
        }
        
        // expr id: S6_092_TEST
        // Tests sailing 6x6 env, using a different starting direction
        if (expr_id == S6_092_TEST) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_TEST_ID;
            int num_trials = 300000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            double default_q_value = -200.0;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in S001_5");
                }
                unordered_map<string,double> alg_params = 
                        {
                            {PARAMS_ID_MENTS_TEMP, temp}, 
                            {PARAMS_ID_MENTS_EPSILON, eps},
                            {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                        };
                if (alg_id == ALG_ID_DENTS) {
                    alg_params.insert_or_assign(PARAMS_ID_DENTS_TEMP, 10.0);
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            unordered_map<string,double> alg_params = 
            {
                {PARAMS_ID_UCT_BIAS, UctManagerArgs::USE_AUTO_BIAS},
                {PARAMS_ID_HMCTS_BUDGET, num_trials},
                {PARAMS_ID_HMCTS_UCT_BUDGET_THRESHOLD, 30},
                {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
            };
            run_ids->push_back(RunID(
                env_id,
                env_instance_id,
                expr_id,
                ALG_ID_HMCTS,
                alg_params,
                num_trials,
                max_trial_length,
                trials_log_delta,
                mc_eval_trials_delta,
                rollouts_per_mc_eval,
                num_repeats,
                num_threads,
                eval_threads));

            return run_ids;
        }

        throw runtime_error("Error in get_run_ids_from_expr_id");
    }

}