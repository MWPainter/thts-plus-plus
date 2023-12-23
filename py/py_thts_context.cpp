#pragma once

#include "py_thts_context.h"

namespace py = pybind11;

namespace thts::py {

    PyThtsContext::PyThtsContext(py::object py_conext) : py_context(py_context) 
    {
    }
}