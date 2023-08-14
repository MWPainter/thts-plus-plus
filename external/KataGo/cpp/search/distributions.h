#pragma once

#include <memory>
#include <random>
#include <vector>
#include <unordered_map>


    /**
     * An abstract class for defining the interface a distribution object should provide. (I.e. we can sample from it)
    */
    template <typename T>
    class BtsDistribution {
        public:
            /**
             * Samples an object of type T from the distribution and returns it.
            */
            virtual T sample(
                std::uniform_int_distribution<int>& int_distr, 
                std::uniform_real_distribution<double>& real_distr, 
                std::ranlux24& rng_gen) = 0;
    };



    /**
     * A discrete uniform distribution
     * 
     * Member variables:
     *      keys: A vector of objects that we want to sample over
    */
    template <typename T>
    class BtsDiscreteUniformDistribution : public BtsDistribution<T> {

        protected:
            std::shared_ptr<std::vector<T>> keys;

        public:
            BtsDiscreteUniformDistribution(std::shared_ptr<std::vector<T>> keys) : keys(keys) {};

            /**
             * Samples a random T uniformly randomly
            */
            virtual T sample(
                std::uniform_int_distribution<int>& int_distr, 
                std::uniform_real_distribution<double>& real_distr, 
                std::ranlux24& rng_gen) 
            {
                int index = int_distr(rng_gen) % keys->size();
                return keys->at(index);
            };
    };



    /**
     * A catagorical distribution
     * 
     * Member variables:
     *      alias_table: 
     *          The alias table to use for O(1) sampling. It is a list of AliasTableEntries
    */
    template <typename T>
    class BtsCategoricalDistribution : public BtsDistribution<T> {

        /**
         * An internal data structure used in the alias table
        */
        struct AliasTableEntry {
            double threshold;
            T first;
            T second;
            AliasTableEntry() : threshold(0.0), first(), second() {};
            AliasTableEntry(
                double threshold, T first) : threshold(threshold), first(first), second(first) {};
            AliasTableEntry(
                double threshold, T first, T second) : threshold(threshold), first(first), second(second) {};
        };

        /**
         * Member variables
        */
        protected:
            std::vector<AliasTableEntry> alias_table;
        
        public:
            /**
             * Constructor
            */
            BtsCategoricalDistribution(std::shared_ptr<std::unordered_map<T,double>> distr) :
                    alias_table(distr->size())
            {
                construct_alias_table(distr);
            };

            /**
             * Samples a random T from the categorical distribution 'distr'
            */
            virtual T sample(
                std::uniform_int_distribution<int>& int_distr, 
                std::uniform_real_distribution<double>& real_distr, 
                std::ranlux24& rng_gen) 
            {
                int rng = int_distr(rng_gen) % alias_table.size();
                AliasTableEntry& alias_entry = alias_table[rng];
                if (1.0 > alias_entry.threshold && alias_entry.threshold > real_distr(rng_gen)) {
                    return alias_entry.first;
                }
                return alias_entry.second;
            };
        
        private:
            /**
             * Initially constructs the 'alias_table' for the initial 'distr'.
             * This will just initially construct the alias table and then call 'reconstruct_alias_table(true)'
            */
            void construct_alias_table(std::shared_ptr<std::unordered_map<T,double>> distr) {
                double sum_weights = 0.0;
                for (std::pair<const T,double>& kv_pair : *distr) {
                    sum_weights += kv_pair.second;
                }

                
                int i = 0;
                int n = distr->size();
                for (std::pair<const T,double> kv_pair : *distr) {
                    double u = kv_pair.second * n / sum_weights;
                    alias_table[i].threshold = u;
                    alias_table[i].first = kv_pair.first;
                    alias_table[i].second = kv_pair.first;
                    i++;
                }

                std::vector<int> large_indices;
                std::vector<int> small_indices;
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
    class BtsMixedDistribution : public BtsDistribution<T> {

        /**
         * Member variables
        */
        protected:
            std::shared_ptr<BtsDistribution<T>> distr_1;
            std::shared_ptr<BtsDistribution<T>> distr_2;
            std::shared_ptr<BtsDistribution<T>> distr_3;
            double w1;
            double w2;
            // double w3; == 1.0 - w1 - w2
        
        public:
            /**
             * Constructor
            */
            BtsMixedDistribution(
                std::shared_ptr<BtsDistribution<T>> distr_1,
                std::shared_ptr<BtsDistribution<T>> distr_2,
                std::shared_ptr<BtsDistribution<T>> distr_3,
                double w1,
                double w2) : 
                    distr_1(distr_1),
                    distr_2(distr_2),
                    distr_3(distr_3),
                    w1(w1),
                    w2(w2) {};

            /**
             * Samples a random T from the mixed distribution
            */
            virtual T sample(
                std::uniform_int_distribution<int>& int_distr, 
                std::uniform_real_distribution<double>& real_distr, 
                std::ranlux24& rng_gen) 
            {
                double rng = real_distr(rng_gen);
                if (rng < w1) {
                    return distr_1->sample(int_distr, real_distr, rng_gen);
                }
                if (rng < w1 + w2) {
                    return distr_2->sample(int_distr, real_distr, rng_gen);
                }
                return distr_3->sample(int_distr, real_distr, rng_gen);
            };
    };