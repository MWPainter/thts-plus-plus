#pragma once

#include "py/py_helper.h"

#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <string>

namespace thts::python {
    namespace py = pybind11;

    /**
     * Enum to specify what is being shared over the shared memory
     */
    enum SharedMemType {
        SMT_none = 0,           // making rpc call with no arguments/return value
        SMT_strings = 1,        // making rpc call with arguments/return value being a vector of strings
        SMT_doubles = 2,        // making rpc call with arguments/return value being a vector of doubles
        SMT_prob_distr = 3,     // making rpc call with arguments/return value being a map from strings to doubles
    };

    /**
     * A wrapper around unix shared memory
     * TODO: long term should add some error checking to make sure that we don't go past shared memory segment
     * TODO: work out how to gracefully shut down server threads if too long
     * 
     * TODO: change name to SharedMemRpc or SharedMemRpcWrapper
     * 
     * N.B. need 'program_unique_filename' in constructor for 'thts::helper::get_unix_key', which needs to be unique 
     * for each instance of thts being run
     * 
     * Member variables:
     *      unix_key:
     *          Key used as an id by unix to refer to semaphore and shared memory
     *      semid:
     *          Unix id for semaphores
     *      shmid:
     *          Unix id for shared memory
     *      shared_mem_size:
     *          The size of the shared memory segment
     *      shared_mem_ptr:
     *          A pointer to the start of the shared memory segment
     *      shared_mem_end_ptr:
     *          A pointer to the end of the shared memory segment
     *      is_server_process:
     *          If we are running in the server process. (Relevant for creating/destroying the shared memory segment, 
     *          which should only be done from the client process).
     * 
     * Member variables (store/read to shared memory):
     *      rpc_id:
     *          An id to indicate what fn the server process should run for the python env 
     *          (takes one of the values in ThtsEnvRpcFn enum from py_multiprocessing_thts_env.h)
     *      value_type:
     *          What type of value is being shared over the shared memory
     *          (takes one of the values in the SharedMemType enum above)
     *      strings:
     *          A public vector of strings, to be written to, or that has been read from shared memory
     *      doubles:
     *          A public vector of doubles, to be written to, or that has been read from shared memory
     *      prob_distr:
     *          A public unordered map from strings to doubles, to be written to, or that has been read from shared 
     *          memory 
     */
    class SharedMemWrapper {
        private:
            key_t unix_key;
            int semid;
            int shmid;
            int shared_mem_size;
            void* shared_mem_ptr;
            void* shared_mem_end_ptr;
            bool is_server_process;

        public:
            int rpc_id;
            int value_type;
            std::shared_ptr<std::vector<std::string>> strings;
            std::shared_ptr<std::vector<double>> doubles;
            std::shared_ptr<std::unordered_map<std::string,double>> prob_distr;

        public:
            SharedMemWrapper(
                std::string& thts_unique_filename, 
                int tid, 
                int shared_mem_size_in_bytes, 
                bool is_server_process=false);
            virtual ~SharedMemWrapper();

            void breakpoint_add();

            // Called by server process, waits to be signalled to start rpc call
            // sem[0] used to signal start rpc call
            // After server is signalled, first need to read items from shared mem into member vars
            void server_wait_for_rpc_call();

            // Called by server process to send rpc response, assumes args[0] is filled with response
            // sem[1] used to signal end rpc call
            void server_send_rpc_call_result();

            // Assumes rpc_id, num_args, args are filled for rpc call
            // Writes args to shared mem
            // Signal sem[0] to start rpc call
            // Waits on sem[1] to wait for rpc call end
            // Reads results out from shared mem
            void make_rpc_call();

            // Dont wait for response when making kill call
            // TODO: consider if better way to do this?
            // TODO: as providing a general wrapper, might be worth making a blocking option, and have a read_rpc_result for non_blocking rpc calls
            void make_kill_rpc_call();

        private:
            void write_to_shared_mem();
            void write_strings_to_shared_mem();
            void write_doubles_to_shared_mem();
            void write_prob_distr_to_shared_mem();

            void read_from_shared_mem();
            void read_strings_from_shared_mem();
            void read_doubles_from_shared_mem();
            void read_prob_distr_from_shared_mem();
    };
}