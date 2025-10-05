#include "algorithms/common/decaying_temp.h"

namespace thts {

    /**
     * Calling schedule function makes a call to the underlying implementation.
     */
    double Schedule::operator()(double num_visits) 
    {
        return compute_schedule_value(num_visits);
    }

    /**
     * ConstSchedule initialiser
     */
    ConstSchedule::ConstSchedule(double const_val) :
        Schedule(),
        const_val(const_val)
    {
    }

    /**
     * ConstSchedule implementation
     */
    double ConstSchedule::compute_schedule_value(double num_visits) const
    {
        return const_val;
    }

    /**
     * SqrtSchedule initialiser
     */
    SqrtSchedule::SqrtSchedule(double temp_at_zero_visits) :
        Schedule(),
        temp_at_zero_visits(temp_at_zero_visits)
    {
    }

    /**
     * SqrtSchedule implementation
     * f(x) = c/sqrt(1+x),
     * where c == temp_at_zero_visits
     */
    double SqrtSchedule::compute_schedule_value(double num_visits) const
    {
        return temp_at_zero_visits / sqrt(1.0 + num_visits);
    }

    /**
     * LogSchedule initialiser
     */
    LogSchedule::LogSchedule(double temp_at_zero_visits) :
        Schedule(),
        temp_at_zero_visits(temp_at_zero_visits)
    {
    }

    /**
     * SqrtSchedule implementation
     * f(x) = c/log(1+x),
     * where c == temp_at_zero_visits
     */
    double LogSchedule::compute_schedule_value(double num_visits) const
    {
        return temp_at_zero_visits / log(exp(1.0) + num_visits);
    }

    /**
     * LinearSchedule initialiser
     * Initialization of y_intercept and grad from the y_intercept (temp_at_zero_visits) and x_intercept (zero_temp_at) 
     * is explained in .h file.
     */
    LinearSchedule::LinearSchedule(double temp_at_zero_visits, double zero_temp_at) :
        Schedule(),
        y_intercept(temp_at_zero_visits),
        grad(-temp_at_zero_visits/zero_temp_at)
    {
    }

    /**
     * Linear temp schedule function implementation
     * f(x) = max(0, mx + c),
     * where m is gradient and c is y_axis intercept.
    */
    double LinearSchedule::compute_schedule_value(double num_visits) const
    {
        double linear_temp = grad * num_visits + y_intercept;
        if (linear_temp < 0)
        {
            return 0.0;
        }
        return linear_temp;
    }
}