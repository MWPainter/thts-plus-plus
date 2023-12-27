

#include "py/py_thts_context.h"

namespace py = pybind11;

namespace thts::python {

    PyThtsContext::PyThtsContext(py::object py_context) : py_context(py_context) 
    {
    }
}