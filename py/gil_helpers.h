#pragma once

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>


namespace thts::python::helpers {
    namespace py = pybind11;

    /**
     * Gil Lock Guard
     * Aquires the lock at creation by making a gil_scoped_acquire
     * Makes a gil_scoped_release in destructor
     * When Lock Guard goes out of scope, then gil_scoped_{aquire,release} are also destructed
    */
    class GilReenterantLockGuard {
        protected:
            static std::unique_ptr<std::mutex> lock;
            static int ref_count;
            std::unique_ptr<py::gil_scoped_acquire> gil_acq;

        public:
            GilReenterantLockGuard();
            ~GilReenterantLockGuard();
            GilReenterantLockGuard(GilReenterantLockGuard&) = delete;
            GilReenterantLockGuard(GilReenterantLockGuard&&) = delete;
    };
    
};