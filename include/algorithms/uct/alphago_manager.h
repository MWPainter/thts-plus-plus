#pragma once

#include "puct_manager.h"

#include <random>
#include <vector>


/**
 * Dirichlet distribution implementation from: https://github.com/gcant/dirichlet-cpp/blob/master/dirichlet.h
*/
namespace thts {
    template <class RNG>
    class DirichletDistribution{
        public:
            DirichletDistribution(const std::vector<double>&);
            void set_params(const std::vector<double>&);
            std::vector<double> get_params();
            std::vector<double> operator()(RNG&);
        private:
            std::vector<double> alpha;
            std::vector<std::gamma_distribution<>> gamma;
    };

    template <class RNG>
    DirichletDistribution<RNG>::DirichletDistribution(const std::vector<double>& alpha){
        set_params(alpha);
    }

    template <class RNG>
    void DirichletDistribution<RNG>::set_params(const std::vector<double>& new_params){
        alpha = new_params;
        std::vector<std::gamma_distribution<>> new_gamma(alpha.size());
        for (unsigned int i=0; i<alpha.size(); ++i){
            std::gamma_distribution<> temp(alpha[i], 1);
            new_gamma[i] = temp;
        }
        gamma = new_gamma;
    }

    template <class RNG>
    std::vector<double> DirichletDistribution<RNG>::get_params(){
        return alpha;
    }

    template <class RNG>
    std::vector<double> DirichletDistribution<RNG>::operator()(RNG& generator){
        std::vector<double> x(alpha.size());
        double sum = 0.0;
        for (unsigned int i=0; i<alpha.size(); ++i){
            x[i] = gamma[i](generator);
            sum += x[i];
        }
        for (double &xi : x) xi = xi/sum;
        return x;
    }
}


namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct AlphaGoManagerArgs : public PuctManagerArgs {
        static constexpr double dirichlet_noise_coeff_default = 0.25;
        static constexpr double dirichlet_noise_param_default = 0.03;

        double dirichlet_noise_coeff;
        double dirichlet_noise_param;

        AlphaGoManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            PuctManagerArgs(thts_env),
            dirichlet_noise_coeff(dirichlet_noise_coeff_default),
            dirichlet_noise_param(dirichlet_noise_param_default) {}

        virtual ~AlphaGoManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for PUCT algorithms.
     * 
     * Member variables:
     *      dirichlet_noise_coeff: THe coefficient to use for dirichlet noise at root node vs prior
     *      dirichlet_noise_param: The parameter to use for the dirichlet distribution
     */
    class AlphaGoManager : public PuctManager {
        public:
            double dirichlet_noise_coeff;
            double dirichlet_noise_param;

            AlphaGoManager(AlphaGoManagerArgs& args) :
                PuctManager(args),
                dirichlet_noise_coeff(args.dirichlet_noise_coeff),
                dirichlet_noise_param(args.dirichlet_noise_param) {};
            
            /**
             * Helper to get dirichlet sample
            */
            std::vector<double> sample_dirichlet(int dimension) {
                std::vector<double> alpha(dimension, dirichlet_noise_param);
                DirichletDistribution<std::mt19937> distr(alpha);
                return distr(real_gen);
            }

    };
}