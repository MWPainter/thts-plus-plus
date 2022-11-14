#include "helper_templates.h"

#include <cstddef>
#include <float.h>
#include <functional>
#include <sstream>


namespace thts::helper {
    using namespace std;

    /**
     * Adapted from boost: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
     */
    template <typename T>
    size_t hash_combine(const size_t cur_hash, const T& v) {
        hash<T> hash_fn;
        return cur_hash ^ (hash_fn(v) + 0x9e3779b9 + (cur_hash << 6) + (cur_hash >> 2));
    }

    /**
     * Select max value with randomly breaking ties
     */
    template <typename T, typename NumericT>
    T get_max_key_break_ties_randomly(unordered_map<T,NumericT>& map, ThtsManager& thts_manager) {
        NumericT best_val = -DBL_MAX;
        vector<T> best_keys;
        for (pair<const T,NumericT> pr : map) {
            NumericT val = pr.second;
            if (val < best_val) continue;
            if (val > best_val) {
                best_keys = vector<T>();
                best_val = val;
            }
            const T key = pr.first;
            best_keys.push_back(key);
        }

        int indx = thts_manager.get_rand_int(0,best_keys.size());
        return best_keys[indx];
    }

    /**
     * Sampling from distribution
     */
    template <typename T>
    T sample_from_distribution(unordered_map<T,double>& distribution, ThtsManager& thts_manager, bool normalised=true) {
        double sum_weights = 1.0
        if (!normalised) {
            sum_weights = 0.0
            for (auto pr : distribution) {
                sum_weights += pr.second;
            }
        }

        double rand_val = thts_manager.get_rand_uniform();
        double running_sum = 0.0;
        for (pair<T,double> pr : distribution) {
            running_sum += pr.second / sum_weights;
            if (rand_val < running_sum) {
                return pr.first;
            }
        }

        throw "Error in sampling if get here. Did you mean to set normalised=false in sampling from distribution?"
    }


    /**
     * Printing vectors
     */
    template <typename T>
    string vector_pretty_print_string(const vector<T>& vec) {
        stringstream ss;
        ss << "[";
        bool first_iter = true;
        for (const T& val : vec) {
            if (!first_iter) {
                ss << ",";
            } else {
                first_iter = false;
            }
            ss << val;
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
        for (const pair<const K, V>& pr : mp) {
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