#include "algorithms/common/decaying_temp.h"

namespace thts {
    /**
     * A common implementation for working out a decayed temperature 
     */
    double compute_decayed_temp(double init_temp, int num_visits, double min_temp) {
        if (num_visits < 1) num_visits = 1;
        double decayed_temp = init_temp / sqrt(num_visits);
        //double decayed_temp = init_temp / log(num_visits+2);
        if (decayed_temp < min_temp) return min_temp;
        return decayed_temp;
    }
}