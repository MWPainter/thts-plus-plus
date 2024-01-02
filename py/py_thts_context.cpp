

#include "py/py_thts_context.h"

#include "py/gil_helpers.h"

namespace py = pybind11;

namespace thts::python {

    PyThtsContext::PyThtsContext(py::object _py_context) : py_context() 
    {
        thts::python::helpers::GilReenterantLockGuard lg();
        py_context = _py_context;
    }
}