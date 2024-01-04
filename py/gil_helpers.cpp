#include "py/gil_helpers.h"

using namespace std;
namespace py = pybind11;

namespace thts::python::helpers {
    
    // Init static variables
    int GilReenterantLockGuard::ref_count = 0;
    unique_ptr<mutex> GilReenterantLockGuard::lock = make_unique<mutex>();

    // Constructor, grabs gil (using ptr) and increases ref count
    // GilReenterantLockGuard::GilReenterantLockGuard(bool delay_locking_gil) : gil_acq(nullptr) {
    //     if (!delay_locking_gil) lock_gil();
    // }
    GilReenterantLockGuard::GilReenterantLockGuard() : gil_acq(nullptr) {
        lock_gil();
    }

    // Lock the gil
    // Note:
    // If hold GilReenterantLockGuard::lock while trying to aquire gil
    // Then can get deadlock
    // When another thread has gil, but cant aquire GilReenterantLockGuard::lock (in destructor)
    // Solution = it's safe to construct gil_scoped_acquire without GilReenterantLockGuard::lock
    // Recall that we're using this to manage the *global* gil (it's the g)
    // So these objects never try to make gil_acq more than once
    void GilReenterantLockGuard::lock_gil() {
        GilReenterantLockGuard::lock->lock();
        GilReenterantLockGuard::ref_count++;
        GilReenterantLockGuard::lock->unlock();
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