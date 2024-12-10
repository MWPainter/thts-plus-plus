#pragma once

#include "py/mo_gym_multiprocessing_thts_env.h"


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    /** 
     * Gym
     */
    class TimedMoGymMultiprocessingThtsEnv : public MoGymMultiprocessingThtsEnv {

        protected:

        public:
            /**
             * Constructor
             */
            TimedMoGymMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string& thts_unique_filename,
                std::string& gym_env_id,
                bool is_server_process=false);

            /**
             * Private copy constructor to implement 
            */
            TimedMoGymMultiprocessingThtsEnv(TimedMoGymMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Override id so "py_env_server" program can identify it needs to use this env.
             */
            virtual std::string get_multiprocessing_env_type_id() override;

            /**
             * Override get_mo_reward so can add an additional time cost to it
             */
            virtual Eigen::ArrayXd get_mo_reward(
                std::shared_ptr<const PyState> state, 
                std::shared_ptr<const PyAction> action,
                ThtsEnvContext& ctx) const override;
    };
}