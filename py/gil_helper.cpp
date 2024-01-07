#include "py/gil_helper.h"

using namespace std;
namespace py = pybind11;





// namespace thts::python::helper {
    
//     // Init static variables
//     bool GilReenterantLockGuard::using_subinterpreters = false;
//     int GilReenterantLockGuard::ref_count = 0;
//     unique_ptr<mutex> GilReenterantLockGuard::ref_count_lock = make_unique<mutex>();
//     PyGILState_STATE GilReenterantLockGuard::gstate;
//     unordered_map<thread::id,unique_ptr<mutex>> GilReenterantLockGuard::ref_count_lock_map;
//     unordered_map<thread::id,int> GilReenterantLockGuard::ref_count_map;
//     unordered_map<thread::id,PyGILState_STATE> GilReenterantLockGuard::gstate_map;

//     // Constructor
//     // Sets up / gets variables for this thread
//     // Lock to protect ref count
//     // Increases ref count for this thread
//     // Releases ref count lock before aqcuiring gil
//     // Needs to release before locking gil
//     // Otherwise can get stuck when constructor has GilReenterantLockGuard::ref_count_lock, but destructor has GIL
//     // Solve by releasing GilReenterantLockGuard::ref_count_lock around lock_gil call
//     // And using a local variable copy of 'ref_count'
//     GilReenterantLockGuard::GilReenterantLockGuard(bool _force) {
//         force = _force;
//         if (using_subinterpreters && !force) return;

//         ensure_variables_exist();

//         thread::id tid = get_thread_id();
//         mutex& _ref_count_lock = get_ref_count_lock(tid);
//         int& _ref_count = get_ref_count(tid);

//         _ref_count_lock.lock();
//         _ref_count++;
//         int local_ref_count = _ref_count; 
//         _ref_count_lock.unlock();

//         if (local_ref_count == 1) {
//             PyGILState_STATE state = lock_gil();
//             set_gstate(tid, state);
//         }
        
//     }

//     // Get threads variables 
//     // Destructor, decreases ref count, releases gil
//     GilReenterantLockGuard::~GilReenterantLockGuard() {
//         if (using_subinterpreters && !force) return;

//         thread::id tid = get_thread_id();
//         mutex& _ref_count_lock = get_ref_count_lock(tid);
//         int& _ref_count = get_ref_count(tid);

//         std::lock_guard<std::mutex> lg(_ref_count_lock);
//         _ref_count--;
//         if (_ref_count == 0) {
//             PyGILState_STATE state = get_gstate(tid);
//             unlock_gil(state);
//         }
//     }

//     // Increment refs + call py (static function)
//     PyGILState_STATE GilReenterantLockGuard::lock_gil() {
//         return PyGILState_Ensure();
//     }

//     // Increment refs + call py (static function)
//     void GilReenterantLockGuard::unlock_gil(PyGILState_STATE gstate) {
//         PyGILState_Release(gstate);
//     }

//     void GilReenterantLockGuard::ensure_variables_exist() {
//         if (!using_subinterpreters) {
//             return;
//         }
//         thread::id tid = get_thread_id();
//         if (ref_count_lock_map.contains(tid)) { 
//             return;
//         }
//         ref_count_lock_map.emplace(tid,make_unique<mutex>());
//         ref_count_map[tid] = 0;
//     }

//     thread::id GilReenterantLockGuard::get_thread_id() {
//         return std::this_thread::get_id();
//     }

//     mutex& GilReenterantLockGuard::get_ref_count_lock(thread::id tid) {
//         return (!using_subinterpreters) ? *ref_count_lock : *ref_count_lock_map[tid];
//     }

//     int& GilReenterantLockGuard::get_ref_count(thread::id tid) {
//         return (!using_subinterpreters) ? ref_count : ref_count_map[tid];
//     }

//     PyGILState_STATE GilReenterantLockGuard::get_gstate(thread::id tid) {
//         return (!using_subinterpreters) ? gstate : gstate_map[tid];
//     }

//     void GilReenterantLockGuard::set_gstate(thread::id tid, PyGILState_STATE state) {
//         if (!using_subinterpreters) {
//             gstate = state;
//         } else {
//             gstate_map[tid] = state;
//         }
//     }




//     // -------------------------------------------------------------------------
//     // Old GIL lock guard using pybind11 stuff
//     // -------------------------------------------------------------------------

//     // // Lock the gil (pybind)
//     // // Note:
//     // // If hold GilReenterantLockGuard::lock while trying to aquire gil
//     // // Then can get deadlock
//     // // When another thread has gil, but cant aquire GilReenterantLockGuard::lock (in destructor)
//     // // Solution = it's safe to construct gil_scoped_acquire without GilReenterantLockGuard::lock
//     // // Recall that we're using this to manage the *global* gil (it's the g)
//     // // So these objects never try to make gil_acq more than once
//     // void GilReenterantLockGuard::lock_gil_pybind11() {
//     //     GilReenterantLockGuard::ref_count_lock->lock();
//     //     GilReenterantLockGuard::ref_count++;
//     //     GilReenterantLockGuard::ref_count_lock->unlock();
//     //     gil_acq = std::make_unique<py::gil_scoped_acquire>();
//     // }

//     // // Unlock gil
//     // void GilReenterantLockGuard::unlock_gil_pybind11() {
//     //     py::gil_scoped_release rel;
//     // }
// }