#pragma once

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>


namespace thts::python::helper {
    namespace py = pybind11;

    /**
     * Gil Lock Guard
     * Constructing the GilReenterantLockGuard will act as a re-enterant lock guard around the GIL
     * 
     * Static methods provide direct access to locking and unlocking the GIL (just providing a central area for GIL 
     * logic really), and do NOT alter the state in this class at all, they are just wrappers around the CPython 
     * interface.
     * 
     * Member variables:
     *      using_subinterpreters (static): 
     *          If using subinterpreters, then dont want to be locking gil in the threads, so these operations become 
     *          no-ops. 
     * 
     *      ref_count_lock (static):
     *          Lock to protect ref_count
     *      ref_count (static):
     *          Number of GilReenterantLockGuards that have 'locked' the GIL
     *      gstate (static):
     *          CPython GIL state object, used to (un)lock the gil
     * 
     *      ++ map versions of each, for when using subinterpreters == true
     *      
    */
    class GilReenterantLockGuard {
        public:
            static bool using_subinterpreters;

        protected:
            static std::unique_ptr<std::mutex> ref_count_lock;
            static int ref_count;
            static PyGILState_STATE gstate;

            static std::unordered_map<std::thread::id,std::unique_ptr<std::mutex>> ref_count_lock_map;
            static std::unordered_map<std::thread::id,int> ref_count_map;
            static std::unordered_map<std::thread::id,PyGILState_STATE> gstate_map;

            bool force;

        public:
            GilReenterantLockGuard(bool force=false);
            ~GilReenterantLockGuard();
            static PyGILState_STATE lock_gil();
            static void unlock_gil(PyGILState_STATE state);
            // void lock_gil_pybind11();
            // void unlock_gil_pybind11();
            GilReenterantLockGuard(GilReenterantLockGuard&) = delete;
            GilReenterantLockGuard(GilReenterantLockGuard&&) = delete;

        // Private setters and getters for state, which works out if should use static versions, or versions in static 
        // maps using 'using_subinterpreters'
        private:
            void ensure_variables_exist();
            std::thread::id get_thread_id();
            std::mutex& get_ref_count_lock(std::thread::id tid);
            int& get_ref_count(std::thread::id tid);
            PyGILState_STATE get_gstate(std::thread::id tid);
            void set_gstate(std::thread::id tid, PyGILState_STATE state);
    };
    
};