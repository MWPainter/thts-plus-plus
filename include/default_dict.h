#pragma once

#include "thts_manager.h"
#include "thts_types.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace thts::helper {
    /**
     * Simple default dict, wrapper around unordered_map type
    */
    template <typename Key, typename T> 
    class unordered_map_with_default { 
    public: 
        T default_value;
        std::unordered_map<Key,T> map;

        unordered_map_with_default(T default_value);
        T& operator[](const Key & key);
    }; 
}

#include "default_dict.cc"