#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>
#include <string>

namespace thts::python {
    namespace py = pybind11;

    /**
     * A wrapper around the pickle module that can be used in a multi-threaded environment
     */
    class PickleWrapper {
        private:
            std::mutex pickle_lock;
            std::shared_ptr<py::module_> py_pickle_module;
            std::shared_ptr<py::object> py_pickle_dumps_fn;
            std::shared_ptr<py::object> py_pickle_loads_fn;

        public:
            PickleWrapper();
            virtual ~PickleWrapper();

            std::string serialise(py::object& py_obj);
            py::object deserialise(std::string& serialised_py_obj_str);
    };
}