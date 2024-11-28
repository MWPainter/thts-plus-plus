#pragma once

#include "py/py_helper.h"

#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <string>

namespace thts::python {
    namespace py = pybind11;

    /**
     * A wrapper around unix shared memory
     * TODO: long term should add some error checking to make sure that we don't go past shared memory segment
     * TODO: work out how to gracefully shut down server threads if too long
     * 
     * TODO: change name to SharedMemRpc or SharedMemRpcWrapper
     * 
     * Member variables:
     *      unix_key:
     *          Key used as an id by unix to refer to semaphore and shared memory
     *      semid:
     *          Unix id for semaphores
     *      shmid:
     *          Unix id for shared memory
     * 
     * Member variables (store/read to shared memory):
     *      rpc_id:
     *          An id to indicate what fn the server process should run for the python env 
     *      num_args:
     *          The number of args to pass to the rpc call
     *      args:
     *          Vector of args as string objects (assumes that they are searlised )
     */
    class SharedMemWrapper {
        private:
            key_t unix_key;
            int semid;
            int shmid;
            int shared_mem_size;
            void* shared_mem_ptr;
            bool is_server_process;

        public:
            int rpc_id;
            int num_args;
            std::string args[3];

        public:
            SharedMemWrapper(int tid, int shared_mem_size_in_bytes, bool is_server_process=false);
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
            void read_from_shared_mem();
            void write_to_shared_mem();
    };
}