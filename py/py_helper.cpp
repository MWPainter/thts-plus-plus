#include "py/py_helper.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace thts::python::helper {
    using namespace std;
    namespace py = pybind11;

    /**
     * 
    */
    key_t get_unix_key() {
        return get_unix_key(0);
    }

    key_t get_unix_key(int tid) {
        return ftok(filesystem::current_path().c_str(), tid+1);
    }

    /**
     * Initialise unix semaphores for Unix Multiprocessing
     * 
     * Adapted from: https://beej.us/guide/bgipc/html/split/semaphores.html#semaphores
     * 
     * In our usecase the main thread will set all of this up before any child processes are spawned, so we've 
     * simplified it
    */
    int init_sem(key_t key, int num_sems, bool is_server_process)  /* key from ftok() */
    {
        // create semaphores (if not server process), just get them if server process
        int flags = 0666;
        if (!is_server_process) {
            flags |= IPC_CREAT | IPC_EXCL;
        }
        int semid = semget(key, num_sems, flags);
        if (semid < 0) {
            stringstream ss;
            ss << "Error creating filesystem semaphores (try running 'ipcrm -v -a' to clear unix semaphores and "
               << "rerunning), errno: " << errno;
            throw runtime_error(ss.str());
        }

        // If getting semaphores from server process, they should be initialised already
        if (is_server_process) {
            return semid;
        }

        // Initialise semaphores to a value of one
        struct sembuf sb;
        sb.sem_op = 1; 
        sb.sem_flg = 0;

        for(sb.sem_num = 0; sb.sem_num < num_sems; sb.sem_num++) { 
            /* do a semop() to "free" the semaphores. */
            /* this sets the sem_otime field, as needed below. */
            if (semop(semid, &sb, 1) == -1) {
                int e = errno;
                semctl(semid, 0, IPC_RMID); /* clean up */
                errno = e;
                stringstream ss;
                ss << "Error in initialising filesystem semaphores, errno: " << errno;
                throw runtime_error(ss.str());
            }
        }
        
        return semid;
    }

    /**
     * Updating sem helper
    */
    void update_sem(int semid, int sem_num, int delta) {
        struct sembuf sb;
        sb.sem_num = sem_num;
        sb.sem_op = delta;
        sb.sem_flg = 0;
        semop(semid, &sb, 1);
    }

    /**
     * Acquire the sem
    */
    void acquire_sem(int semid, int sem_num) {
        update_sem(semid, sem_num, -1);
    }

    /**
     * Release sem
    */
    void release_sem(int semid, int sem_num) {
        update_sem(semid, sem_num, 1);
    }

    /**
     * Removes sem from os
    */
    void destroy_sem(int semid) {
        semctl(semid, 0, IPC_RMID);
    }

    /**
     * Makes a piece of shared memory
     * Returns shmid used to identify the piece of shared memory
    */
    int init_shared_mem(key_t key, int size_in_bytes, bool is_server_process) {
        int flags = 0644;
        if (!is_server_process) {
            flags |= IPC_CREAT;
        }
        int shmid = shmget(key, size_in_bytes, flags);
        if (shmid == -1) {
            stringstream ss;
            ss << "Error making shared memory, errno: " << errno;
            throw runtime_error(ss.str());
        }
        return shmid;
    }

    /**
     * Gets ptr to shared memory given shared memory id 'shmid'
    */
    void* get_shared_mem_ptr(int shmid) {
        void* data = shmat(shmid, (void*)0, 0);
        if (data == (void*)(-1)) {
            stringstream ss;
            ss << "Error attaching to shared memory, errno: " << errno;
            throw runtime_error(ss.str());
        }
        return data;
    }

    /**
     * detach
     */
    void detach_shared_mem(void* data_ptr)
    {
        shmdt(data_ptr);
    }

    /**
     * Destroy shared memory associated with 'shmid'
     * Also need to detach from the shared memory segment first
    */
    void destroy_shared_mem(int shmid) {
        shmctl(shmid, IPC_RMID, NULL);
    }
}