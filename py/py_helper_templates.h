#pragma once

#include <memory>
#include <mutex>

namespace thts::python::helper {
    /**
     * Given shared pointers to resources r1 and r2 with associated locks l1 and l2 will lock l1,l2 without deadlock
     * 
     * Uses ordering based on the values of the pointers
     * 
     * Checks for if &r1 == &r2 and if so only locks l1 as it is likely l1==l2 
     * 
     * Args:
     *      rp1: pointer to r1
     *      l1: mutex for r1
     *      rp2: pointer to r2
     *      l2: mutex for r2
     */
    template <typename T>
    void ordered_lock(
        const std::shared_ptr<T> rp1, 
        std::recursive_mutex& l1, 
        const std::shared_ptr<T>rp2, 
        std::recursive_mutex& l2);
}

#include "py/py_helper_templates.cc"