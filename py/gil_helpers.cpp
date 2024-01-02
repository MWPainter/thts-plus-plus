#include "py/gil_helpers.h"

using namespace std;
namespace py = pybind11;

namespace thts::python::helpers {
    
    // Init static variables
    int GilReenterantLockGuard::ref_count = 0;
    unique_ptr<mutex> GilReenterantLockGuard::lock = make_unique<mutex>();

    // Constructor, grabs gil (using ptr) and increases ref count
    GilReenterantLockGuard::GilReenterantLockGuard() {
        std::lock_guard<std::mutex> lg(*GilReenterantLockGuard::lock);
        GilReenterantLockGuard::ref_count++;
        gil_acq = std::make_unique<py::gil_scoped_acquire>();
    }

    // Destructor, decreases ref count, 
    GilReenterantLockGuard::~GilReenterantLockGuard() {
        std::lock_guard<std::mutex> lg(*GilReenterantLockGuard::lock);
        GilReenterantLockGuard::ref_count--;
        if (GilReenterantLockGuard::ref_count == 0) {
            py::gil_scoped_release rel;
        }
    }
}