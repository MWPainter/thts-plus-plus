#include "py/py_helper_templates.h"


namespace thts::python::helper {
    using namespace std;

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
}