#include "helper_templates.h"

#include <cstddef>
#include <functional>
#include <sstream>

using namespace std;

namespace thts::helper {
    /**
     * Adapted from boost: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
     */
    template <typename T>
    size_t hash_combine(const size_t cur_hash, const T& v) {
        hash<T> hash_fn;
        return cur_hash ^ (hash_fn(v) + 0x9e3779b9 + (cur_hash << 6) + (cur_hash >> 2));
    }

    /**
     * Printing vectors
     */

    template <typename T>
    string vector_pretty_print_string(const vector<T>& vec) {
        stringstream ss;
        ss << "[";
        bool first_iter = true;
        for (const T val : vec) {
            if (!first_iter) {
                ss << ",";
            } else {
                first_iter = false;
            }
            ss << val << ",";
        }
        ss << "]";
        return ss.str();
    }

    /**
     * Printing maps
     */
    template <typename K, typename V>
    string unordered_map_pretty_print_string(const unordered_map<K,V>& mp, string delimiter) {
        stringstream ss;
        ss << "{";
        bool first_iter = true;
        for (pair<const K, V> pr : mp) {
            if (!first_iter) {
                ss << ",";
            } else {
                first_iter = false;
            }
            ss << pr.first << delimiter << pr.second ;
        }
        ss << "}";
        return ss.str();
    }
}