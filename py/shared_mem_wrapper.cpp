#include "py/shared_mem_wrapper.h"

#include <iostream>

using namespace std;
namespace py = pybind11;

namespace thts::python {
    SharedMemWrapper::SharedMemWrapper(int tid, int shared_mem_size_in_bytes, bool is_server_process) :
        unix_key(thts::python::helper::get_unix_key(tid)),
        semid(thts::python::helper::init_sem(unix_key, 2, is_server_process)),
        shmid(thts::python::helper::init_shared_mem(unix_key, shared_mem_size_in_bytes, is_server_process)),
        shared_mem_size(shared_mem_size_in_bytes),
        shared_mem_ptr(thts::python::helper::get_shared_mem_ptr(shmid)),
        shared_mem_end_ptr(nullptr),
        is_server_process(is_server_process),
        rpc_id(),
        value_type(),
        strings(nullptr),
        doubles(nullptr),
        prob_distr(nullptr)
    {
        // acquire the sems by default (if client process)
        if (!is_server_process) {
            thts::python::helper::acquire_sem(semid, 0);
            thts::python::helper::acquire_sem(semid, 1);
        }

        // Compute end pointer
        char* beg_shm_ptr = (char*) shared_mem_ptr;
        char* end_shm_ptr = beg_shm_ptr + shared_mem_size_in_bytes;
        shared_mem_end_ptr = (void*) end_shm_ptr;
    };

    SharedMemWrapper::~SharedMemWrapper() {
        thts::python::helper::detach_shared_mem(shared_mem_ptr);
        if (!is_server_process) {
            thts::python::helper::destroy_sem(semid);
            thts::python::helper::destroy_shared_mem(shmid);
        }
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

    // When killing the server, it wont respond, so dont wait for a response
    void SharedMemWrapper::make_kill_rpc_call() {
        write_to_shared_mem();
        thts::python::helper::release_sem(semid, 0);
    }

    /**
     * Read whatever values have been put shared memory to the public members of this object
     * And clear any values that are currently stored in this object
     */
    void SharedMemWrapper::read_from_shared_mem() 
    {
        strings.reset();
        doubles.reset();
        prob_distr.reset();

        int* shm_rpc_id_ptr = (int*) shared_mem_ptr;
        rpc_id = *shm_rpc_id_ptr;
        int* shm_value_type_ptr = shm_rpc_id_ptr + 1;
        value_type = *shm_value_type_ptr;

        switch (value_type)
        {
            case SMT_none:
                return;
            case SMT_strings:
                return read_strings_from_shared_mem();
            case SMT_doubles:
                return read_doubles_from_shared_mem();
            case SMT_prob_distr:
                return read_prob_distr_from_shared_mem();
            default:
                throw runtime_error("Trying to read from shared memory and encountered invalid value type.");
        }
    }

    /**
     * See write_strings_to_shared_mem, just doing the reverse operation
     */
    void SharedMemWrapper::read_strings_from_shared_mem()
    {
        // Read the size of vector, make a vector to store it, and get pointer to first string (length)
        size_t* shm_vector_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        size_t vector_size = *shm_vector_size_ptr;
        strings = make_shared<vector<string>>();
        strings->reserve(vector_size);
        size_t* shm_string_len_ptr = shm_vector_size_ptr + 1;

        // Read strings
        for (size_t i=0; i<vector_size; i++) {
            size_t string_len = *shm_string_len_ptr;
            char* shm_string_ptr = (char*) (shm_string_len_ptr + 1);
            strings->push_back(std::string(shm_string_ptr, shm_string_ptr+string_len));
            // Update pointer for next string
            shm_string_len_ptr = (size_t*) (shm_string_ptr + string_len+1);
        }
    }

    /**
     * See write_doubles_to_shared_mem, just doing the reverse operation
     */
    void SharedMemWrapper::read_doubles_from_shared_mem()
    {
        // Read the size of vector, make a vector to store it, and get pointer to first string (length)
        size_t* shm_vector_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        size_t vector_size = *shm_vector_size_ptr;
        doubles = make_shared<vector<double>>();
        doubles->reserve(vector_size);
        double* shm_double_ptr = (double*) (shm_vector_size_ptr + 1);

        // Read strings
        for (size_t i=0; i<vector_size; i++) {
            doubles->push_back(*shm_double_ptr);
            shm_double_ptr += 1;
        }
    }

    /**
     * See write_prob_distr_to_shared_mem, just doing the reverse operation
     */
    void SharedMemWrapper::read_prob_distr_from_shared_mem()
    {
        // Read the size of map, make a map to store it, and get pointer to first string (length)
        size_t* shm_map_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        size_t map_size = *shm_map_size_ptr;
        prob_distr = make_shared<unordered_map<string,double>>();
        prob_distr->reserve(map_size);
        size_t* shm_string_len_ptr = shm_map_size_ptr + 1;

        // Read strings, double pairs and put in map
        for (size_t i=0; i<map_size; i++) {
            size_t string_len = *shm_string_len_ptr;
            char* shm_string_ptr = (char*) (shm_string_len_ptr + 1);
            string key = std::string(shm_string_ptr, shm_string_ptr+string_len);
            double* shm_double_ptr = (double*) (shm_string_ptr + string_len+1);
            double value = *shm_double_ptr;
            prob_distr->insert_or_assign(key, value);
            // Update pointer for next string
            shm_string_len_ptr = (size_t*) (shm_double_ptr + 1);
        }
    }



    /**
     * Write whatever values have been put in the public member variables to shared memory
     */
    void SharedMemWrapper::write_to_shared_mem() 
    {
        int* shm_rpc_id_ptr = (int*) shared_mem_ptr;
        (*shm_rpc_id_ptr) = rpc_id;
        int* shm_value_type_ptr = shm_rpc_id_ptr + 1;
        (*shm_value_type_ptr) = value_type;

        switch (value_type)
        {
            case SMT_none:
                return;
            case SMT_strings:
                return write_strings_to_shared_mem();
            case SMT_doubles:
                return write_doubles_to_shared_mem();
            case SMT_prob_distr:
                return write_prob_distr_to_shared_mem();
            default:
                throw runtime_error("Trying to write invalid value type in shared memory wrapper.");
        }
    }

    /**
     * Beginning of values is 2*size(integer) into shared memory (because rpc_id and value type are there)
     * The vector is encoded as follows:
     *      - first is size_t giving the size of the vector
     *      - each string is then encoded contiguously after
     *      - each string is encoded as its length, followed by the char* data
     */
    void SharedMemWrapper::write_strings_to_shared_mem() 
    {   
        // Write size of vector, and get pointer to first string (length)
        size_t* shm_vector_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        (*shm_vector_size_ptr) = strings->size();
        size_t* shm_string_len_ptr = shm_vector_size_ptr + 1;

        // Write strings
        for (size_t i=0; i<strings->size(); i++) {
            size_t string_len = strings->at(i).size();
            (*shm_string_len_ptr) = string_len;
            char* shm_string_ptr = (char*) (shm_string_len_ptr + 1);
            memcpy(shm_string_ptr, strings->at(i).c_str(), string_len+1);
            // Update pointer for next string
            shm_string_len_ptr = (size_t*) (shm_string_ptr + string_len+1);
        }

        // Check haven't written over the end of shared memory
        // Probably would get segfault, but in the case its a small violation and we're at risk of being clobbered
        if (shm_string_len_ptr >= shared_mem_end_ptr) {
            throw runtime_error(
                "Written over the end of shared memory, increase the size of the shared memory segments being used.");
        }
    }

    /**
     * Beginning of values is 2*size(integer) into shared memory (because rpc_id and value type are there)
     * The vector is encoded as follows:
     *      - first is size_t giving the size of the vector
     *      - then the doubles are all stored contiguously
     */
    void SharedMemWrapper::write_doubles_to_shared_mem() 
    {   
        // Write size of vector, and get pointer to first double
        size_t* shm_vector_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        (*shm_vector_size_ptr) = doubles->size();
        double* shm_double_ptr = (double*) (shm_vector_size_ptr + 1);

        // Write doubles
        for (size_t i=0; i<doubles->size(); i++) {
            (*shm_double_ptr) = doubles->at(i);
            shm_double_ptr += 1;
        }

        // Check haven't written over the end of shared memory
        // Probably would get segfault, but in the case its a small violation and we're at risk of being clobbered
        if (shm_double_ptr >= shared_mem_end_ptr) {
            throw runtime_error(
                "Written over the end of shared memory, increase the size of the shared memory segments being used.");
        }
    }

    /**
     * Beginning of values is 2*size(integer) into shared memory (because rpc_id and value type are there)
     * The map is encoded as follows:
     *      - first is size_t giving the size of the map
     *      - each item is then encoded contiguously after
     *      - first each string is encoded by its length and then its char* data
     *      - after the double value that the string maps to is stored
     * 
     * This is basically the same as write_string_to_shared_mem, but adds a double after each 
     */
    void SharedMemWrapper::write_prob_distr_to_shared_mem() 
    {   
        // Write size of vector, and get pointer to first string (length)
        size_t* shm_vector_size_ptr = (size_t*) (((int*)shared_mem_ptr) + 2);
        (*shm_vector_size_ptr) = strings->size();
        size_t* shm_string_len_ptr = shm_vector_size_ptr + 1;

        // Write strings and doubles
        for (pair<string,double> pair : *prob_distr) {
            size_t string_len = pair.first.size();
            (*shm_string_len_ptr) = string_len;
            char* shm_string_ptr = (char*) (shm_string_len_ptr + 1);
            memcpy(shm_string_ptr, pair.first.c_str(), string_len+1);
            double* shm_double_ptr = (double*) (shm_string_ptr + string_len+1);
            (*shm_double_ptr) = pair.second;
            // Update pointer for next string, double pair
            shm_string_len_ptr = (size_t*) (shm_double_ptr + 1);
        }

        // Check haven't written over the end of shared memory
        // Probably would get segfault, but in the case its a small violation and we're at risk of being clobbered
        if (shm_string_len_ptr >= shared_mem_end_ptr) {
            throw runtime_error(
                "Written over the end of shared memory, increase the size of the shared memory segments being used.");
        }
    }
}