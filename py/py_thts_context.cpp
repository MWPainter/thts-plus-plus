

#include "py/py_thts_context.h"

using namespace std;
namespace py = pybind11;

namespace thts::python {

    PyThtsContext::PyThtsContext(shared_ptr<py::object> _py_context) : py_context(_py_context) 
    {
    } 

    PyThtsContext::~PyThtsContext()
    {
        py_context.reset();
    } 
}