#pragma once

#include "py/mo_py_multiprocessing_thts_env.h"


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    /** 
     * Gym
     */
    class MoGymMultiprocessingThtsEnv : public MoPyMultiprocessingThtsEnv {

        protected:
            std::string gym_env_id;

        public:
            /**
             * Constructor
             */
            MoGymMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string& gym_env_id);

            /**
             * Private copy constructor to implement 
            */
            MoGymMultiprocessingThtsEnv(MoGymMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;
    };
}