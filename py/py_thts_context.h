#pragma once

#include "thts_env_context.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>

namespace thts::python {
    namespace py = pybind11;

    /**
     * A subclass of ThtsEnvContext that adds a public python object to be used as a context in python portions of the 
     * code.
     * 
     * Protecting python variables:
     * - contexts are per thread, so dont need to be protected from concurrent acess
     * 
     * Member variables:
     *      py_context_lock: A lock to protect 
     *      py_context: A python object to be used as part of the context
     */
    class PyThtsContext : public ThtsEnvContext {
        public:
            std::mutex py_context_lock;
            std::shared_ptr<py::object> py_context;

            PyThtsContext(std::shared_ptr<py::object> _py_context);
            virtual ~PyThtsContext();
    };
}