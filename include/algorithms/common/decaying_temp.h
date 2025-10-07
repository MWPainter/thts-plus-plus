#pragma once

#include <cmath>

namespace thts {
    /**
     * Schedule interface.
    */
    class Schedule {

        public:
            Schedule() = default;
            virtual ~Schedule() = default;

        protected:
            /**
             * Implementation of the schedule function/operator() for subclasses to override and implement.
             */
            virtual double compute_schedule_value(double num_visits) const = 0;

        public:
            /**
             * Call this schedule function, returns a "scheduleed coefficient" >= 0 with respect to the inputs.
             * Inputs is num_visits nominally to indicate that it will usually be the number of visits to a node.
             * But input is double to be more general.
             */
            double operator()(double num_visits);
    };

    /**
     * A defualt schedule function that returns a constant value. 
     */
    class ConstSchedule : public Schedule {

        private:
            double const_val;

        public:
            ConstSchedule(double const_val);
            virtual ~ConstSchedule() = default;

        protected:
            virtual double compute_schedule_value(double num_visits) const override;

    };

    /**
     * Inverse square root temp schedule function
     * f(x) = c/sqrt(1+dx),
     * where c == temp_at_zero_visits
     * and d == decay_rate_coeff
    */
    class SqrtSchedule : public Schedule {

        private:
            double temp_at_zero_visits;
            double decay_rate_coeff;

        public:
            SqrtSchedule(double temp_at_zero_visits, double decay_rate_coeff);
            virtual ~SqrtSchedule() = default;

        protected:
            virtual double compute_schedule_value(double num_visits) const override;
    };

    /**
     * Inverse log temp schedule function
     * f(x) = c/log(1+dx),
     * where c == temp_at_zero_visits
     * and d == decay_rate_coeff
    */
    class LogSchedule : public Schedule {

        private:
            double temp_at_zero_visits;
            double decay_rate_coeff;

        public:
            LogSchedule(double temp_at_zero_visits, double decay_rate_coeff);
            virtual ~LogSchedule() = default;

        protected:
            virtual double compute_schedule_value(double num_visits) const override;
    };

    /**
     * Linear temp schedule function
     * f(x) = max(0, mx + c)
     * 
     * Can be initialised by the x_axis and y_axis intercepts for interprebility
     * - the y intercept (temp_at_zero_visits) is the initial temperature (temp_at_zero_visits)
     * - the x intercept (zero_temp_at) is how many trials or which we have a temp > 0
     * 
     * Computing y intercept (c) and gradient (m) from this: 
     *      m * 0 + c = temp_at_zero_visits, gives c = temp_at_zero_visits
     *      m * zero_temp_at + c = 0, gives m = -temp_at_zero_visits/zero_temp_at
     */
    class LinearSchedule : public Schedule {

        private:
            double y_intercept;
            double grad;

        public:
            LinearSchedule(double temp_at_zero_visits, double zero_temp_at);
            virtual ~LinearSchedule() = default;

        protected:
            virtual double compute_schedule_value(double num_visits) const override;
    };
}