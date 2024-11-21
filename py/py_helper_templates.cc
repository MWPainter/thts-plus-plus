#include "py/py_helper_templates.h"


namespace thts::python::helper {
    using namespace std;
    namespace py = pybind11;

    template <typename T>
    void ordered_lock(const shared_ptr<T> rp1, recursive_mutex& l1, const shared_ptr<T>rp2, recursive_mutex& l2) {
        if (rp1 == rp2) {
            l1.lock();
            return;
        }
        if (rp1 < rp2) {
            l1.lock();
            l2.lock();
        } else {
            l2.lock();
            l1.lock();
        }
    }

    template <typename T>
    T call_py_getter(std::shared_ptr<py::object> py_obj_ptr, std::string thunk_name) {
        py::gil_scoped_acquire acquire;
        return py_obj_ptr->attr(py::str(thunk_name))().cast<T>();
    }
}