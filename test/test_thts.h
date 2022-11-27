#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "thts.h"

#include "test_thts_env.h"
#include "test_thts_nodes.h"
#include "thts_decision_node.h"
#include "thts_manager.h"

#include <chrono>
#include <memory>
#include <thread>

namespace thts_test {
    using namespace std;
    using namespace thts;
    /**
     * Mocker for testing the thread pool part of the ThtsPool
     */
    class MockThtsPool_PoolTesting : public ThtsPool {
        public:
            MockThtsPool_PoolTesting(
                shared_ptr<ThtsManager> thts_manager, 
                shared_ptr<ThtsDNode> root_node, 
                int num_threads=1) :
                    ThtsPool(thts_manager, root_node, num_threads)  {};

            MOCK_METHOD(void, run_thts_trial, (int), (override));

            // Setup for being able to test 'work_left'
            void mock_work_left_scenario(
                int trials_remaining,
                chrono::time_point<chrono::system_clock> start_time,
                chrono::duration<double> max_run_time) 
            {
                this->trials_remaining = trials_remaining;
                this->start_time = start_time;
                this->max_run_time = max_run_time;
            }
    };

    /**
     * Mocker for testing the thread pool does run concurrently
     */
    class MockThtsPool_DurationTrialPoolTesting : public ThtsPool {
        private:
            int trial_duration_ms;

        public:
            /**
             * Constructor takes a duration for trials to be mocked with
             */
            MockThtsPool_DurationTrialPoolTesting(
                int trial_duration_ms,
                shared_ptr<ThtsManager> thts_manager, 
                shared_ptr<ThtsDNode> root_node, 
                int num_threads=1) :
                    ThtsPool(thts_manager, root_node, num_threads), 
                    trial_duration_ms(trial_duration_ms) {};

            /**
             * Mock running a trial by sleeping for 'trial_duration' seconds.
             */
            void run_thts_trial(int num_trials_remaining) override {
                this_thread::sleep_for(chrono::milliseconds(trial_duration_ms));
            }

            /**
             * Getter for mutex so can lock it in tests
             */
            mutex& get_work_left_lock() {
                return work_left_lock;
            }
    };

    /**
     * Subclass of ThtsPool that makes the protected functions public
     */ 
    class PublicThtsPool : public ThtsPool {
        public:
            PublicThtsPool(
                shared_ptr<ThtsManager> thts_manager=nullptr, 
                shared_ptr<ThtsDNode> root_node=nullptr, 
                int num_threads=1) :
                    ThtsPool(thts_manager, root_node, num_threads) {};
            virtual ~PublicThtsPool() = default;
            virtual bool work_left() { return ThtsPool::work_left(); };
            virtual bool should_continue_selection_phase(
                shared_ptr<ThtsDNode> cur_node, bool new_decision_node_created_this_trial) 
            {
                return ThtsPool::should_continue_selection_phase(cur_node, new_decision_node_created_this_trial);
            }
            void run_selection_phase(
                vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                vector<double>& rewards, 
                ThtsEnvContext& context)
            {
                ThtsPool::run_selection_phase(nodes_to_backup, rewards, context);
            }
            void run_backup_phase(
                vector<pair<shared_ptr<ThtsDNode>,shared_ptr<ThtsCNode>>>& nodes_to_backup, 
                vector<double>& rewards, 
                ThtsEnvContext& context)
            {
                ThtsPool::run_backup_phase(nodes_to_backup, rewards, context);
            }
            virtual void run_thts_trial(int num_trials_remaining) 
            {
                ThtsPool::run_thts_trial(num_trials_remaining); 
            };
            virtual void worker_fn() { ThtsPool::worker_fn(); };

            // public anyway:
            // virtual void join();
            // virtual void run_trials(
            //     int max_trials=numeric_limits<int>::max(), 
            //     double max_time=numeric_limits<double>::max(), 
            //     bool blocking=true);   
    };

    /**
     * Mocker for TestThtsDNode
     */
    class MockThtsDNode : public TestThtsDNode {
        public:
            MockThtsDNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep) :
                    TestThtsDNode(thts_manager,state,decision_depth,decision_timestep) {}

            MOCK_METHOD(bool, is_leaf, (), (const, override));
            MOCK_METHOD(void, visit_itfc, (ThtsEnvContext&), (override));
            MOCK_METHOD(shared_ptr<const Action>, select_action_itfc, (ThtsEnvContext&), (override));
            MOCK_METHOD(
                void, 
                backup_itfc, 
                (const vector<double>&,const vector<double>&,const double,const double,ThtsEnvContext&), 
                (override));
            MOCK_METHOD(
                shared_ptr<ThtsCNode>, 
                get_child_node_itfc, 
                (shared_ptr<const Action>),
                (const, override));
            MOCK_METHOD(int, get_num_children, (), (const, override));

            void set_heuristic_value(double val) { heuristic_value = val; }
    };

    /**
     * Mocker for TestThtsCNode
     */
    class MockThtsCNode : public TestThtsCNode {
        public:
            MockThtsCNode(
                shared_ptr<ThtsManager> thts_manager,
                shared_ptr<const State> state,
                shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep) :
                    TestThtsCNode(thts_manager,state,action,decision_depth,decision_timestep) {}

            MOCK_METHOD(void, visit_itfc, (ThtsEnvContext&), (override));
            MOCK_METHOD(
                shared_ptr<const Observation>, sample_observation_itfc, (ThtsEnvContext&), (override));
            MOCK_METHOD(
                void, 
                backup_itfc, 
                (const vector<double>&,const vector<double>&,const double,const double,ThtsEnvContext&), 
                (override));
            MOCK_METHOD(
                shared_ptr<ThtsDNode>, 
                get_child_node_itfc, 
                (shared_ptr<const Observation>),
                (const, override));
            MOCK_METHOD(int, get_num_children, (), (const, override));
    };

    /**
     * Mocker for TestThtsEnv, so can mock get_reward
     */
    class MockTestThtsEnv : public TestThtsEnv {
        public:
            MockTestThtsEnv(int grid_size) : TestThtsEnv(grid_size) {};

            MOCK_METHOD(
                double, 
                get_reward_itfc, 
                (shared_ptr<const State>,
                    shared_ptr<const Action>,
                    shared_ptr<const Observation>),
                (const, override));
    };

    /**
     * Mocker for PublicThtsPool (above) that mocks the independently tested 'should_continue_selection_phase'
     */
    class MockPublicThtsPool : public PublicThtsPool {
        public:
            MockPublicThtsPool(
                shared_ptr<ThtsManager> thts_manager=nullptr, 
                shared_ptr<ThtsDNode> root_node=nullptr, 
                int num_threads=1) :
                    PublicThtsPool(thts_manager, root_node, num_threads) {};

            MOCK_METHOD(bool, should_continue_selection_phase, (shared_ptr<ThtsDNode>,bool), (override));
    };
}

