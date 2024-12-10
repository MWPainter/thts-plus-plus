#include "main/val.h"

#include <iostream>

// test 0
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"
#include "py/test_env.h"
#include "thts.h"
#include "thts_env_context.h"

// test 1
#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"

// test 2
#include "mo/czt_manager.h"
#include "mo/czt_decision_node.h"
#include "mo/mo_mc_eval.h"
#include "mo/mo_thts.h"
#include "test/mo/test_mo_thts_env.h"

// test 3
#include "mo/chmcts_manager.h"
#include "mo/chmcts_decision_node.h"

// test 4
#include "mo/smt_bts_manager.h"
#include "mo/smt_bts_decision_node.h"

// test 5
#include "mo/smt_dents_manager.h"
#include "mo/smt_dents_decision_node.h"

// test 6 + 7
#include "py/pickle_wrapper.h"
#include "py/py_multiprocessing_thts_env.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>


using namespace std;
using namespace thts;
using namespace thts::python;
namespace py = pybind11;

namespace thts {
    /**
     * Test 0
     * - sanity, core lib
     */
    void core_lib_test() {
        int env_size = 5;
        double stay_prob = 0.2;
        int num_threads = 1;
        int print_tree_depth = 2;

        shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
        UctManagerArgs manager_args(grid_env);
        manager_args.seed = 60415;
        manager_args.max_depth = env_size * 4;
        manager_args.mcts_mode = false;
        shared_ptr<UctManager> manager = make_shared<UctManager>(manager_args);
        shared_ptr<UctDNode> root_node = make_shared<UctDNode>(manager, grid_env->get_initial_state_itfc(), 0, 0);
        ThtsPool thts_pool(manager, root_node, num_threads);
        thts_pool.run_trials(100);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    }

    /**
     * Test 1
     * - sanity, core lib
     */
    void multi_thread_test() {
        int env_size = 5;
        double stay_prob = 0.2;
        int num_threads = 4;
        int print_tree_depth = 2;

        shared_ptr<ThtsEnv> grid_env = make_shared<TestThtsEnv>(env_size, stay_prob);
        DentsManagerArgs manager_args(grid_env);
        manager_args.seed = 60415;
        manager_args.max_depth = env_size * 4;
        manager_args.mcts_mode = false;
        manager_args.temp = 1.0;
        shared_ptr<DentsManager> manager = make_shared<DentsManager>(manager_args);
        shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, grid_env->get_initial_state_itfc(), 0, 0);
        ThtsPool thts_pool(manager, root_node, num_threads);
        thts_pool.run_trials(100);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;
    }

    /**
     * Test 2
     * - CZT, C++ envs
     */
    void czt_cpp_env_and_mo_mc_eval_test() {

        // params
        double bias = 4.0;
        int num_backups_before_allowed_to_split = 10;

        int walk_len = 5;
        double stay_prob = 0.2;

        int num_trials = 100;
        int print_tree_depth = 2;
        int num_threads = 4;

        // Setup env 
        shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

        // Make thts manager 
        shared_ptr<CztManagerArgs> args = make_shared<CztManagerArgs>(thts_env);
        args->seed = 60415;
        args->max_depth = walk_len * 4;
        args->mcts_mode = false;
        args->bias = bias;
        args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
        args->num_threads = num_threads;
        args->num_envs = num_threads; 
        shared_ptr<CztManager> manager = make_shared<CztManager>(*args);

        // Run search
        shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
        shared_ptr<CztDNode> root_node = make_shared<CztDNode>(manager, init_state, 0, 0);
        shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
        thts_pool->run_trials(num_trials);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;

        // Also Test out Mo MC Eval
        int num_eval_rollouts = 250;

        shared_ptr<EvalPolicy> policy = make_shared<EvalPolicy>(root_node, thts_env, manager);
        MoMCEvaluator mo_mc_eval(
            policy,  
            manager->max_depth,
            manager,
            Eigen::ArrayXd::Zero(2)-walk_len,
            Eigen::ArrayXd::Zero(2)-0.5*walk_len);
            
        mo_mc_eval.run_rollouts(num_eval_rollouts, num_threads);

        cout << "CZT evaluations from MoMCEval." << endl;
        cout << "Mean MO return." << endl;
        cout << mo_mc_eval.get_mean_mo_return() << endl;
        cout << "Mean MO ctx return." << endl;
        cout << mo_mc_eval.get_mean_mo_ctx_return() << endl;
        cout << "Mean MO normalised ctx return." << endl;
        cout << mo_mc_eval.get_mean_mo_normalised_ctx_return() << endl;
    }

    /**
     * Test 3
     * - CHMCTS, C++ envs
     * 
     * *alarm*
     * haz memo leak, should debug
     * 
     * So apparently glp_init_env() allocates memory that never gets released
     * Which is called from glpk, which is called from lemon
     * In convex_hull.cc
     * Seems that it's a consistent ~5kb that gets leaked, regardless of number of trials
     * So dont think this is the main issue
     * Although it is annoying -.-
     * 
     * Tried to call glp_free_env(), as discussed here: https://stackoverflow.com/questions/21785221/cleanest-way-to-make-glpk-clean-up-at-the-program-termination 
     * But couldnt get it working, hmph
     */
    void chmcts_cpp_env_test() {

        // params
        double bias = 4.0;
        int num_backups_before_allowed_to_split = 10;

        int walk_len = 5;
        double stay_prob = 0.2;

        int num_trials = 100; // 1000;
        int print_tree_depth = 2;
        int num_threads = 4; // 16;

        // Setup env 
        shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

        // Make thts manager 
        shared_ptr<ChmctsManagerArgs> args = make_shared<ChmctsManagerArgs>(thts_env);
        args->seed = 60415;
        args->max_depth = walk_len * 4;
        args->mcts_mode = false;
        args->bias = bias;
        args->num_backups_before_allowed_to_split = num_backups_before_allowed_to_split;
        args->num_threads = num_threads;
        args->num_envs = num_threads; 
        shared_ptr<ChmctsManager> manager = make_shared<ChmctsManager>(*args);

        // Run search
        shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
        shared_ptr<ChmctsDNode> root_node = make_shared<ChmctsDNode>(manager, init_state, 0, 0);
        shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
        thts_pool->run_trials(num_trials);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;

    }

    /**
     * Test 4
     * - SMBTS, C++ envs
     * 
     * *alarm*
     * haz memo leak, should debug
     * 
     * Ok, memo leak fixed
     * NGV's in simplex maps were creating cycles of shared_ptr objects, which means they would never get freed
     * Made sure that they get reset in the simplex map destructor
     */
    void smbts_cpp_env_test() {

        // params
        int walk_len = 5;
        double stay_prob = 0.2;

        int num_trials = 100;
        int print_tree_depth = 2;
        int num_threads = 4;

        // Setup env 
        shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

        // Make thts manager 
        Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(2) - walk_len * 4;
        shared_ptr<SmtBtsManagerArgs> args = make_shared<SmtBtsManagerArgs>(thts_env, default_val);
        args->seed = 60415;
        args->max_depth = walk_len * 4;
        args->mcts_mode = false;
        args->num_threads = num_threads;
        args->num_envs = num_threads; 
        args->simplex_node_l_inf_thresh = 0.01;
        shared_ptr<SmtBtsManager> manager = make_shared<SmtBtsManager>(*args);

        // Run search
        shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
        shared_ptr<SmtBtsDNode> root_node = make_shared<SmtBtsDNode>(manager, init_state, 0, 0);
        shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
        thts_pool->run_trials(num_trials);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;

    }

    /**
     * Test 5
     * - SMDENTS, C++ envs
     * 
     * *alarm*
     * haz memo leak, should debug
     * 
     * Was just the same leak as SMBTS, now fixed
     */
    void smdents_cpp_env_test() {

        // params
        int walk_len = 5;
        double stay_prob = 0.2;

        int num_trials = 100;
        int print_tree_depth = 2;
        int num_threads = 4;

        // Setup env 
        shared_ptr<thts::test::TestMoThtsEnv> thts_env = make_shared<thts::test::TestMoThtsEnv>(walk_len, stay_prob);

        // Make thts manager 
        Eigen::ArrayXd default_val = Eigen::ArrayXd::Zero(2) - walk_len * 4;
        shared_ptr<SmtDentsManagerArgs> args = make_shared<SmtDentsManagerArgs>(thts_env, default_val);
        args->seed = 60415;
        args->max_depth = walk_len * 4;
        args->mcts_mode = false;
        args->num_threads = num_threads;
        args->num_envs = num_threads; 
        args->simplex_node_l_inf_thresh = 0.01;
        shared_ptr<SmtDentsManager> manager = make_shared<SmtDentsManager>(*args);

        // Run search
        shared_ptr<const State> init_state = thts_env->get_initial_state_itfc();
        shared_ptr<SmtDentsDNode> root_node = make_shared<SmtDentsDNode>(manager, init_state, 0, 0);
        shared_ptr<ThtsPool> thts_pool = make_shared<MoThtsPool>(manager, root_node, num_threads);
        thts_pool->run_trials(num_trials);

        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;

    }

    /**
     * Test 6 + 7
     * - Python Env, single objective
     * 
     * For #6, seems to be a consistent 1Mb of still reachable/possibly lost memory, coming from CPython
     * - should be fine, right?
     * 
     * And #7 is the same
     * 
     * Again, tried running this with 1000 trials, and 5000 trials, and it deosnt seem to grow
     */
    void python_single_obj_env_test(int num_threads) {
        // python iterpreter + release gil
        py::scoped_interpreter py_interpreter;
        py::gil_scoped_release rel;

        // params
        int env_size = 3;
        double stay_prob = 0.2;
        int num_trials = 100;
        int print_tree_depth = 2;

        // Make py env (making a py::object of python thts env, and pass into constructor)
        shared_ptr<ThtsEnv> thts_env;
        {
            py::gil_scoped_acquire acq;
            string module_name = "test_env"; 
            string class_name = "PyTestThtsEnv";
            py::dict kw_args;
            kw_args["grid_size"] = env_size;
            kw_args["stay_prob"] = stay_prob;
            shared_ptr<PickleWrapper> pickle_wrapper = make_shared<PickleWrapper>();
            string thts_unique_filename = "/";
            thts_env = make_shared<PyMultiprocessingThtsEnv>(
                pickle_wrapper, thts_unique_filename, module_name, class_name, make_shared<py::dict>(kw_args));
        } 

        // Make thts manager with the py env 
        DentsManagerArgs args(thts_env);
        args.seed = 60415;
        args.max_depth = env_size * 4;
        args.mcts_mode = false;
        args.temp = 1.0;
        args.num_threads = num_threads;
        args.num_envs = num_threads;
        shared_ptr<DentsManager> manager = make_shared<DentsManager>(args);

        // Setup python servers
        for (int i=0; i<args.num_envs; i++) {
            PyMultiprocessingThtsEnv& py_mp_env = *dynamic_pointer_cast<PyMultiprocessingThtsEnv>(manager->thts_env(i));
            py_mp_env.start_python_server(i);
        }
        
        shared_ptr<EstDNode> root_node = make_shared<EstDNode>(manager, thts_env->get_initial_state_itfc(), 0, 0);
        shared_ptr<ThtsPool> bts_pool = make_shared<ThtsPool>(manager, root_node, num_threads);

        // Run thts trials (same as c++)
        bts_pool->run_trials(num_trials);

        // Print out a tree (same as c++)
        cout << "Tree from test:" << endl;
        cout << root_node->get_pretty_print_string(print_tree_depth) << endl;

    }

    /**
     * Entrypoint
     */
    void run_valgrind_debugging(int test_no) {
        switch (test_no) {
            case 0:
                return core_lib_test();
            case 1:
                return multi_thread_test();
            case 2:
                return czt_cpp_env_and_mo_mc_eval_test();
            case 3:
                return chmcts_cpp_env_test();
            case 4:
                return smbts_cpp_env_test();
            case 5:
                return smdents_cpp_env_test();
            case 6:
                return python_single_obj_env_test(1);
            case 7:
                return python_single_obj_env_test(4);
            default:
                throw runtime_error("No test for test no " + to_string(test_no));
        }
    }
}