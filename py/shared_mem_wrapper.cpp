#include "py/shared_mem_wrapper.h"

using namespace std;
namespace py = pybind11;

namespace thts::python {
    SharedMemWrapper::SharedMemWrapper(int tid, int shared_mem_size_in_bytes) :
        unix_key(thts::python::helper::get_unix_key(tid)),
        semid(thts::python::helper::init_sem(unix_key, 2)),
        shmid(thts::python::helper::init_shared_mem(unix_key,shared_mem_size_in_bytes)),
        shared_mem_size(shared_mem_size_in_bytes),
        shared_mem_ptr(thts::python::helper::get_shared_mem_ptr(shmid)),
        rpc_id(),
        num_args(),
        args()    
    {
        // acquire the sems by default
        thts::python::helper::acquire_sem(semid, 0);
        thts::python::helper::acquire_sem(semid, 1);
    };

    SharedMemWrapper::~SharedMemWrapper() {
        thts::python::helper::destroy_sem(semid);
        thts::python::helper::destroy_shared_mem(shmid);
    };

    // Called by server process, waits to be signalled to start rpc call
    // sem[0] used to signal start rpc call
    // After server is signalled, first need to read items from shared mem into member vars
    void SharedMemWrapper::server_wait_for_rpc_call() {
        thts::python::helper::acquire_sem(semid, 0);
        read_from_shared_mem();
    }

    // Called by server process to send rpc response, assumes args[0] is filled with response
    // sem[1] used to signal end rpc call
    void SharedMemWrapper::server_send_rpc_call_result() {
        write_to_shared_mem();
        thts::python::helper::release_sem(semid, 1);
    }

    // Assumes rpc_id, num_args, args are filled for rpc call
    // Writes args to shared mem
    // Signal sem[0] to start rpc call
    // Waits on sem[1] to wait for rpc call end
    // Reads results out from shared mem
    void SharedMemWrapper::make_rpc_call() {
        write_to_shared_mem();
        thts::python::helper::release_sem(semid, 0);
        thts::python::helper::acquire_sem(semid, 1);
        read_from_shared_mem(); 
    }

    void SharedMemWrapper::read_from_shared_mem() {
        int* shm_rpc_id = (int*) shared_mem_ptr;
        rpc_id = *shm_rpc_id;
        int* shm_num_args = shm_rpc_id + 1;
        num_args = *shm_num_args;
        int* shm_arg_len = (int*)(shm_num_args + 1);
        char* shm_arg = (char*)(shm_arg_len + 1);
        for (int i=0; i<num_args; i++) {
            int n = *shm_arg_len;
            args[i] = std::string(shm_arg, shm_arg+n);
            shm_arg_len = (int*)(shm_arg + n+1);
            shm_arg = (char*)(shm_arg_len + 1);
        }
    }

    void SharedMemWrapper::write_to_shared_mem() {
        int* shm_rpc_id = (int*) shared_mem_ptr;
        (*shm_rpc_id) = rpc_id;
        int* shm_num_args = shm_rpc_id + 1;
        (*shm_num_args) = num_args;
        int* shm_arg_len = (int*)(shm_num_args + 1);
        char* shm_arg = (char*)(shm_arg_len + 1);
        for (int i=0; i<num_args; i++) {
            int n = args[i].size();
            (*shm_arg_len) = n;
            memcpy(shm_arg, args[i].c_str(), n+1);
            shm_arg_len = (int*)(shm_arg + n+1);
            shm_arg = (char*)(shm_arg_len + 1);
        }
    }
}