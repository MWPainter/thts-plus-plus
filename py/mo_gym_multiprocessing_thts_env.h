#pragma once

#include "py/mo_py_multiprocessing_thts_env.h"


namespace thts::python {
    // PyBind
    using namespace thts;
    namespace py = pybind11;

    // ID to identify this env for server processes
    static std::string MOGYM_ENV_SERVER_ID = "mo_gym_mp_env";

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
                std::string& thts_unique_filename,
                std::string& gym_env_id,
                bool is_server_process=false);

            /**
             * Private copy constructor to implement 
            */
            MoGymMultiprocessingThtsEnv(MoGymMultiprocessingThtsEnv& other);

            /**
             * Clone - virtual copy constructor idiom
            */
            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Override id so "py_env_server" program can identify it needs to use this env.
             */
            virtual std::string get_multiprocessing_env_type_id() override;
            virtual void fill_multiprocessing_args(std::vector<std::string>& args, int tid) override;
    };
}