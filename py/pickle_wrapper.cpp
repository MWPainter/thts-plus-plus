#include "py/pickle_wrapper.h"

#include <iostream>

using namespace std;
namespace py = pybind11;

namespace thts::python {

    PickleWrapper::PickleWrapper() : 
        py_pickle_module(), 
        py_pickle_dumps_fn(),
        py_pickle_loads_fn()
    {
        py::gil_scoped_acquire acquire;
        py_pickle_module = make_shared<py::module_>(py::module_::import("pickle"));
        py_pickle_dumps_fn = make_shared<py::object>(py_pickle_module->attr("dumps"));
        py_pickle_loads_fn = make_shared<py::object>(py_pickle_module->attr("loads"));
    } 

    PickleWrapper::~PickleWrapper()
    {
        py::gil_scoped_acquire acquire;
        py_pickle_module.reset();
        py_pickle_dumps_fn.reset();
        py_pickle_loads_fn.reset();
    } 

    string PickleWrapper::serialise(py::object& py_obj)
    {
        py::gil_scoped_acquire acquire;
        py::object py_serialised_obj = (*py_pickle_dumps_fn)(py_obj);
        return py_serialised_obj.cast<string>();
    }

    py::object PickleWrapper::deserialise(string& serialised_py_obj_str) 
    {
        py::gil_scoped_acquire acquire;
        py::object py_serialised_obj = py::bytes(serialised_py_obj_str);
        return (*py_pickle_loads_fn)(py_serialised_obj);
    }
}