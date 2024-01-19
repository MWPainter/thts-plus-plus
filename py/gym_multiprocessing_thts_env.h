#pragma once

#include "py/py_multiprocessing_thts_env.h"


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    /** 
     * Gym
     */
    class GymMultiprocessingThtsEnv : public PyMultiprocessingThtsEnv {

        protected:
            std::string gym_env_id;

        public:
            /**
             * Constructor
             */
            GymMultiprocessingThtsEnv(
                std::shared_ptr<PickleWrapper> pickle_wrapper,
                std::string& gym_env_id);

            /**
             * Private copy constructor to implement 
            */
            GymMultiprocessingThtsEnv(GymMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;
    };
}



