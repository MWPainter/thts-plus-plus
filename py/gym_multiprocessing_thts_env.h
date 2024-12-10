#pragma once

#include "py/py_multiprocessing_thts_env.h"


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    // ID to identify this env for server processes
    static std::string GYM_ENV_SERVER_ID = "gym_mp_env";

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
                std::string& thts_unique_filename,
                std::string& gym_env_id,
                bool is_server_process=false);

            /**
             * Private copy constructor to implement 
            */
            GymMultiprocessingThtsEnv(GymMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Adds the arguments needed in to run the "py_env_server" program for this env.
             */
            virtual std::string get_multiprocessing_env_type_id() override;
            virtual void fill_multiprocessing_args(std::vector<std::string>& args, int tid) override;
            
    };
}



