#pragma once

#include <memory>
#include <mutex>

#include <pybind11/pybind11.h>

namespace thts::python::helper {
    namespace py = pybind11;
    /**
     * Given shared pointers to resources r1 and r2 with associated locks l1 and l2 will lock l1,l2 without deadlock
     * 
     * Uses ordering based on the values of the pointers
     * 
     * Checks for if &r1 == &r2 and if so only locks l1 as it is likely l1==l2 
     * 
     * Args:
     *      rp1: pointer to r1
     *      l1: mutex for r1
     *      rp2: pointer to r2
     *      l2: mutex for r2
     */
    template <typename T>
    void ordered_lock(
        const std::shared_ptr<T> rp1, 
        std::recursive_mutex& l1, 
        const std::shared_ptr<T>rp2, 
        std::recursive_mutex& l2);

    /**
     * Helper to call a getter function on a python object protected with the gil
     * Useful if want to call a getter as an argument to a C++ function
     * - for example, in constructor of MoPyMultiprocessingThtsEnv, where first used/implemented this for
     */
    template <typename T>
    T call_py_getter(std::shared_ptr<py::object> py_obj_ptr, std::string thunk_name);
}

#include "py/py_helper_templates.cc"