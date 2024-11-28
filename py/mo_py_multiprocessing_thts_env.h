#pragma once

#include "py/py_multiprocessing_thts_env.h"
#include "mo/mo_thts_env.h"

#include <pybind11/pybind11.h>


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    // ID to identify this env for server processes
    static std::string MOPY_ENV_SERVER_ID = "mo_py_mp_env";

    /** 
     * 
     */
    class MoPyMultiprocessingThtsEnv : public PyMultiprocessingThtsEnv, public MoThtsEnv {

        /**
         * Core ThtsEnv implementation functinos.
         */
        public:
            /**
             * Constructor, passing python object directly
             */
            MoPyMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::shared_ptr<py::object> py_thts_env);

            /**
             * Constructor, passing python module name, class name, and constructor args
             */
            MoPyMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string module_name,
                std::string class_name,
                std::shared_ptr<py::dict> constructor_kw_args);

            /**
             * Private copy constructor to implement 
            */
            MoPyMultiprocessingThtsEnv(MoPyMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MoPyMultiprocessingThtsEnv() = default;

            /**
             * Override id so "py_env_server" program can identify it needs to use this env.
             */
            virtual std::string get_multiprocessing_env_type_id() override;
        
        public:

            /**
             * Mo override throw error for single obj
            */
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                ThtsEnvContext& ctx) const override;
            
            /**
             * Mo get reward itfc
             * - implementation takes PyState and PyAction
            */
            virtual Eigen::ArrayXd get_mo_reward(
                std::shared_ptr<const PyState> state, 
                std::shared_ptr<const PyAction> action,
                ThtsEnvContext& ctx) const;
            virtual Eigen::ArrayXd get_mo_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                ThtsEnvContext& ctx) const override;
                
            /**
             * No pycontext, just return MoThtsContext when context get
            */
            virtual std::shared_ptr<ThtsEnvContext> sample_context_itfc(
                int tid, RandManager& rand_manager) const override;


            /**
             * Virtual functions that exist in both children that need to point to right place
            */
            virtual std::shared_ptr<const State> get_initial_state_itfc() const override;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state, ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager, 
                 ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                ThtsEnvContext& ctx) const override;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                 RandManager& rand_manager, 
                 ThtsEnvContext& ctx) const override;
            virtual void reset_itfc() const override;
    };
}



