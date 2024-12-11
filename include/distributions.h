#pragma once

#include "thts_manager.h"

#include "helper_templates.h"
#include "thts_types.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace thts {
    using namespace std;

    /**
     * An abstract class for defining the interface a distribution object should provide. (I.e. we can sample from it)
    */
    template <typename T>
    class Distribution {
        public:
            /**
             * Samples an object of type T from the distribution and returns it.
            */
            virtual T sample(RandManager& rand_manager) = 0;

            /**
             * Gets the current distribution in a map
            */
            virtual shared_ptr<unordered_map<T,double>> get_distr_map() = 0;
    };



    /**
     * A discrete uniform distribution
     * 
     * Member variables:
     *      keys: A vector of objects that we want to sample over
    */
    template <typename T>
    class DiscreteUniformDistribution : public Distribution<T> {

        protected:
            shared_ptr<vector<T>> keys;

        public:
            DiscreteUniformDistribution(shared_ptr<vector<T>> keys) : keys(keys) {};

            /**
             * Samples a random T uniformly randomly
            */
            virtual T sample(RandManager& rand_manager) {
                int index = rand_manager.get_rand_int(0, keys->size());
                return keys->at(index);
            };

            /**
             * Gets the current distribution in a map
            */
            virtual shared_ptr<unordered_map<T,double>> get_distr_map() {
                shared_ptr<unordered_map<T,double>> distr_map = make_shared<unordered_map<T,double>>();
                double n = keys->size();
                for (T& key : *keys) {
                    distr_map->insert_or_assign(key, 1.0/n);
                }
                return distr_map;
            };
            
    };



    /**
     * A catagorical distribution
     * 
     * Member variables:
     *      distr: 
     *          A dictionary mapping from objects of type T to doubles, defining the categorical distribution
     *      use_alias_method: 
     *          A boolean stating if the alias method should be used
     *      reconstruct_alias_table_freq: 
     *          The integer frequency of how often to reconstruct the alias table (with respect to the number of times 
     *          the update function is called). Default value is 1, to update the alias table every time 'update' is 
     *          called.
     *      num_updates: 
     *          An integer keeping track of how many times probabilities have been updated 
     *      alias_table: 
     *          The alias table to use for O(1) sampling. It is a list of AliasTableEntries
    */
    template <typename T>
    class CategoricalDistribution : public Distribution<T> {

        /**
         * An internal data structure used in the alias table
        */
        struct AliasTableEntry {
            double threshold;
            T first;
            T second;
            AliasTableEntry(
                double threshold, T first) : threshold(threshold), first(first), second(first) {};
            AliasTableEntry(
                double threshold, T first, T second) : threshold(threshold), first(first), second(second) {};
        };

        /**
         * Member variables
        */
        protected:
            shared_ptr<unordered_map<T,double>> distr;
            bool use_alias_method;
            int reconstruct_alias_table_freq;
            int num_updates;
            vector<AliasTableEntry> alias_table;
        
        public:
            /**
             * Constructor
            */
            CategoricalDistribution(
                shared_ptr<unordered_map<T,double>> distr, 
                bool use_alias_method=false, 
                int reconstruct_alias_table_freq=1) :
                    distr(distr),
                    use_alias_method(use_alias_method),
                    reconstruct_alias_table_freq(reconstruct_alias_table_freq),
                    num_updates(0),
                    alias_table()
            {
                if (use_alias_method) {
                    construct_alias_table();
                }
            };

            /**
             * Samples a random T from the categorical distribution 'distr'
            */
            virtual T sample(RandManager& rand_manager) {
                if (!use_alias_method) {
                    return helper::sample_from_distribution(*distr, rand_manager, false);
                }

                int rng = rand_manager.get_rand_int(0, distr->size());
                AliasTableEntry& alias_entry = alias_table[rng];
                if (1.0 > alias_entry.threshold && alias_entry.threshold > rand_manager.get_rand_uniform()) {
                    return alias_entry.first;
                }
                return alias_entry.second;
            };

            /**
             * Gets the current distribution in a map
            */
            virtual shared_ptr<unordered_map<T,double>> get_distr_map() {
                return distr;
            };

            /**
             * Updates the probability weight of 'key' in 'distr' to have density 'weight'. 
             * I.e. sets distr[key] = weight.
            */
            void update(T key, double weight){
                distr->insert_or_assign(key, weight);
                num_updates++;
                if (use_alias_method && num_updates == reconstruct_alias_table_freq) {
                    reconstruct_alias_table();
                    num_updates = 0;
                }
            };

            /**
             * Explicitly reconstruct alias table, updating all values in distr
            */
            void reconstruct_alias_table(shared_ptr<unordered_map<T,double>> new_distr) {
                distr = new_distr;
                _reconstruct_alias_table();
            }
        
        private:
            /**
             * Initially constructs the 'alias_table' for the initial 'distr'.
             * This will just initially construct the alias table and then call 'reconstruct_alias_table(true)'
            */
            void construct_alias_table() {
                double sum_weights = 0.0;
                for (pair<const T,double>& kv_pair : *distr) {
                    sum_weights += kv_pair.second;
                }

                int n = distr->size();
                for (pair<const T,double> kv_pair : *distr) {
                    double u = kv_pair.second * n / sum_weights;
                    alias_table.push_back(AliasTableEntry(u, kv_pair.first));
                }

                _reconstruct_alias_table(true);
            };

            /**
             * Reconstructs 'alias_table' for the current 'distr'.
             * This will reset the alias table at the start, unless 'just_constructed == true'
            */
            void _reconstruct_alias_table(bool just_constructed=false) {
                double sum_weights = 0.0;
                for (pair<const T,double>& kv_pair : *distr) {
                    sum_weights += kv_pair.second;
                }

                if (!just_constructed) {
                    int i = 0;
                    int n = distr->size();
                    for (pair<const T,double> kv_pair : *distr) {
                        double u = kv_pair.second * n / sum_weights;
                        alias_table[i].threshold = u;
                        alias_table[i].first = kv_pair.first;
                        alias_table[i].second = kv_pair.first;
                        i++;
                    }
                }

                vector<int> large_indices;
                vector<int> small_indices;
                large_indices.reserve(distr->size());
                small_indices.reserve(distr->size());
                
                for (size_t i = 0; i < alias_table.size(); i++) {
                    if (alias_table[i].threshold > 1.0) {
                        large_indices.push_back(i);
                    } else if (alias_table[i].threshold < 1.0) {
                        small_indices.push_back(i);
                    }
                }

                while (large_indices.size() > 0 && small_indices.size() > 0) {
                    int l_index = large_indices.back();
                    int s_index = small_indices.back();
                    large_indices.pop_back();
                    small_indices.pop_back();
                    
                    alias_table[s_index].second = alias_table[l_index].first;
                    alias_table[l_index].threshold -= 1.0 - alias_table[s_index].threshold;

                    if (alias_table[l_index].threshold > 1.0) {
                        large_indices.push_back(l_index);
                    } else if (alias_table[l_index].threshold < 1.0) {
                        small_indices.push_back(l_index);
                    }
                }

                for (int l_index : large_indices) {
                    alias_table[l_index].threshold = 1.0;
                }

                for (int s_index : small_indices) {
                    alias_table[s_index].threshold = 1.0;
                }
            };
        
    };



    /**
     * Typedef long types in mixed distr
    */
    template<typename T>
    using MixedDistributionDistr = unordered_map<shared_ptr<Distribution<T>>,double>;

    /**
     * A mixed distribution
     * 
     * This represents a mixed distribution over other distributions.
     * 
     * Assumes that the weights of the given distribution sum to 1.0.
     * 
     * Member variables:
     *      distr: 
     *          A dictionary mapping from shared_ptr<Distribution> to doubles, defining the categorical distribution 
     *          over other distributions.
    */
    template <typename T>
    class MixedDistribution : public Distribution<T> {

        /**
         * Member variables
        */
        protected:
            shared_ptr<MixedDistributionDistr<T>> distr;
        
        public:
            /**
             * Constructor
            */
            MixedDistribution(shared_ptr<MixedDistributionDistr<T>> distr) : distr(distr) {};

            /**
             * Samples a random T from the mixed distribution
            */
            virtual T sample(RandManager& rand_manager){
                shared_ptr<Distribution<T>> sampled_distr = helper::sample_from_distribution(*distr, rand_manager);
                return sampled_distr->sample(rand_manager);
            };

            /**
             * Gets the current distribution in a map
            */
            virtual shared_ptr<unordered_map<T,double>> get_distr_map() {
                shared_ptr<unordered_map<T,double>> distr_map = make_shared<unordered_map<T,double>>();
                for (pair<shared_ptr<Distribution<T>>,double> pr : *distr) {
                    shared_ptr<Distribution<T>> sub_distr = pr.first;
                    double prob = pr.second;
                    shared_ptr<unordered_map<T,double>> sub_distr_map = sub_distr->get_distr_map();
                    for (pair<T,double> pr : *sub_distr_map) {
                        T key = pr.first;
                        double weight = pr.second;
                        (*distr_map)[key] += prob * weight;
                    }
                }
                return distr_map;
            }
        
    };
}