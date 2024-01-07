#include "py/py_helper.h"


namespace thts::python::helper {
    using namespace std;
    namespace py = pybind11;
    
    PyGILState_STATE lock_gil() {
        return PyGILState_Ensure();
    }
    
    void unlock_gil(PyGILState_STATE gstate) {
        PyGILState_Release(gstate);
    }
}