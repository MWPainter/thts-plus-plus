#include "thts_env_context.h"

using namespace std;

namespace thts {
    /**
     * Implementation of 'get_value_for_key'
     * 
     * Just try to access the private context map. If there is an error in the access, we want it to be thrown anyway.
     */
    double ThtsEnvContext::get_value_for_key(const string& key) {
        return context.at(key);
    }

    /**
     * Implementation of 'put_value'
     */
    void ThtsEnvContext::put_value(const string& key, double val) {
        context[key] = val;
    }
}