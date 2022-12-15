#pragma once

#include <cmath>

namespace thts {
    /**
     * A common implementation for working out a decayed temperature 
     */
    double get_decayed_temp(double init_temp, int num_visits, double min_temp);
}