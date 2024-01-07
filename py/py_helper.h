#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>

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
}