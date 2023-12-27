#pragma once

#include "thts_env_context.h"

#include <pybind11/pybind11.h>

namespace thts::python {
    namespace py = pybind11;

    /**
     * A subclass of ThtsEnvContext that adds a public python object to be used as a context in python portions of the 
     * code.
     * 
     * Member variables:
     *      py_context: A python object to be used as part of the context
     */
    class PyThtsContext : public ThtsEnvContext {
        public:
            py::object py_context;
            PyThtsContext(py::object init_context);
    };
}