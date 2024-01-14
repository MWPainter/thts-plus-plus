#include "py/pickle_wrapper.h"

#include <iostream>

using namespace std;
namespace py = pybind11;

namespace thts::python {

    PickleWrapper::PickleWrapper() : 
        pickle_lock(), 
        py_pickle_module(), 
        py_pickle_dumps_fn(),
        py_pickle_loads_fn()
    {
        py_pickle_module = make_shared<py::module_>(py::module_::import("pickle"));
        py_pickle_dumps_fn = make_shared<py::object>(py_pickle_module->attr("dumps"));
        py_pickle_loads_fn = make_shared<py::object>(py_pickle_module->attr("loads"));
    } 

    PickleWrapper::~PickleWrapper()
    {
        py_pickle_module.reset();
        py_pickle_dumps_fn.reset();
        py_pickle_loads_fn.reset();
    } 

    string PickleWrapper::serialise(py::object& py_obj)
    {
        pickle_lock.lock();
        py::object py_serialised_obj = (*py_pickle_dumps_fn)(py_obj);
        pickle_lock.unlock();
        return py_serialised_obj.cast<string>();
        // const char* bytes_arr_ptr = PyBytes_AsString(py_serialised_obj.ptr());
        // size_t bytes_arr_len = PyBytes_Size(py_serialised_obj.ptr());
        // return string(bytes_arr_ptr, bytes_arr_ptr+bytes_arr_len);
    }

    py::object PickleWrapper::deserialise(string& serialised_py_obj_str) 
    {
        py::object py_serialised_obj = py::bytes(serialised_py_obj_str);
        lock_guard<mutex> lg(pickle_lock);
        return (*py_pickle_loads_fn)(py_serialised_obj);
    }
}