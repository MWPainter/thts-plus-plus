#pragma once

#include "thts_types.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace thts::helper {
    /**
     * Adds an object into a hash value. Can be used to create a hash for composite objects, by combinding the hashes 
     * of the composed objects.
     * 
     * Args:
     *      cur_hash: A value representing an existing hash value
     *      v: An object that we wish to combine into the hash value
     * 
     * Returns:
     *      A hash value based on the existing 'cur_hash' and additional value 'v'.
     */
    template <typename T>
    std::size_t hash_combine(const std::size_t cur_hash, const T& v);

    /**
     * Helper function for printing vector types to strings. Assumes that the type T can be fed into an ostream.
     * 
     * Args:
     *      vec: The vector to pretty print
     * 
     * Returns:
     *      A string representing a pretty printed version of the vector.
     */
    template <typename T>
    std::string vector_pretty_print_string(const std::vector<T>& vec);


    /**
     * Helper function for printing unordered_map types to strings. Assumes that the types K,V can be fed into an 
     * ostream.
     * 
     * Args:
     *      mp: The map to pretty print
     *      delimiter: A delimiter to use between the keys and values
     * 
     * Returns:
     *      A string representing a pretty printed version of the vector.
     */
    template <typename K, typename V>
    std::string unordered_map_pretty_print_string(const std::unordered_map<K,V>& mp, std::string delimiter=":");
}

#include "helper_templates.cc"