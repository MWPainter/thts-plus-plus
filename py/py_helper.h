#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>

#include <sys/sem.h>
#include <sys/shm.h>

namespace thts::python::helper {
    namespace py = pybind11;
    /**
     * CPython lock gil helper
    */
    PyGILState_STATE lock_gil();
    /**
     * CPython unlock gil helper
    */
    void unlock_gil(PyGILState_STATE gstate);

    /**
     * Get unix keys
     * Basically a wrapper around ftok for our application
    */
    key_t get_unix_key();
    key_t get_unix_key(int tid);

    /**
     * Initialise unix semaphores for Unix Multiprocessing
     * Returns semid used to identify the semaphore set
    */
    int init_sem(key_t key, int num_sems);

    /**
     * Acquire unix semaphore (decrease value by 1)
    */
    void acquire_sem(int semid, int sem_num);

    /**
     * Release unix semaphore (increase value by 1)
    */
    void release_sem(int semid, int sem_num);

    /**
     * Destroy sem
    */
    void destroy_sem(int semid);

    /**
     * Makes a piece of shared memory
     * Returns shmid used to identify the piece of shared memory
    */
    int init_shared_mem(key_t key, int size_in_bytes);

    /**
     * Gets ptr to shared memory given shared memory id 'shmid'
    */
    void* get_shared_mem_ptr(int shmid);

    /**
     * Destroy shared memory associated with 'shmid'
    */
    void destroy_shared_mem(int shmid);
}