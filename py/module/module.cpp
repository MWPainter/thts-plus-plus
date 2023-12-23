#include "test_env.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;


int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(thts, m) {
    // Module docstring
    m.doc() = "python module to access the THTS++ library";

    // This is an example module call for BTS
    // TODO1: add py::object argument to take pass in a custom env
    // TODO2: add custom pybind objects to return (top levels of the) search tree
    m.def("add", &add, "A function that adds two numbers");
};