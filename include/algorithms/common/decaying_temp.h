#pragma once

#include <cmath>

namespace thts {
    /**
     * DecayFn interface.
    */
    class DecayFn {

        public:
            DecayFn() = default;
            virtual ~DecayFn() = default;

        protected:
            virtual double compute_decayed_temp(int num_visits) const = 0;

        public:
            double operator()(int num_visits) 
            {
                return compute_decayed_temp(num_visits);
            }
    };

    /**
     * Inverse square root temp decay function
     * f(m) = 1/sqrt(1+m)
    */
    class SqrtDecayFn : public DecayFn {

        private:
            double temp_at_zero_visits;

        public:
            SqrtDecayFn(double temp_at_zero_visits) :
                DecayFn(),
                temp_at_zero_visits(temp_at_zero_visits)
            {
            }

        protected:
            virtual double compute_decayed_temp(int num_visits) const override
            {
                return temp_at_zero_visits / sqrt(1.0 + num_visits);
            }
    };

    /**
     * Inverse log temp decay function
     * f(m) = 1/log(1+m)
    */
    class LogDecayFn : public DecayFn {

        private:
            double temp_at_zero_visits;

        public:
            LogDecayFn(double temp_at_zero_visits) :
                DecayFn(),
                temp_at_zero_visits(temp_at_zero_visits)
            {
            }

        protected:
            virtual double compute_decayed_temp(int num_visits) const override
            {
                return temp_at_zero_visits / log(exp(1.0) + num_visits);
            }
    };

    /**
     * Linear temp decay function
     * f(x) = max(0, mx + c)
     * 
     * Can be initialised by the x_axis and y_axis intercepts for interprebility
     * - the y intercept is the initial temperature (temp_at_zero_visits)
     * - the x intercept is how many trials we have a temp > 0
     * 
     * c = temp_at_zero_visits
     * As m * zero_temp_at + temp_at_zero_visits = 0, gives m = -temp_at_zero_visits/zero_temp_at
    */
    class LinearDecayFn : public DecayFn {

        private:
            double y_intercept;
            double grad;

        public:
            LinearDecayFn(double temp_at_zero_visits, double zero_temp_at) :
                DecayFn(),
                y_intercept(temp_at_zero_visits),
                grad(-temp_at_zero_visits/zero_temp_at)
            {
            }

        protected:
            virtual double compute_decayed_temp(int num_visits) const override
            {
                double linear_temp = grad * num_visits + y_intercept;
                if (linear_temp < 0)
                {
                    return 0.0;
                }
                return linear_temp;
            }
    };
}