#include "thts_env.h"

using namespace std;

namespace thts {
    /**
     * Default implementation of 'sample_context'
     * 
     * Returns an (default constructed) ThtsEnvContext, which is really just a wrapper around an empty map. It's useful 
     * to return this type so we can subclass it, rather than forcing Thts algorithms to use a specific map for a 
     * context.
     */
    ThtsEnvContext ThtsEnv::sample_context_itfc(shared_ptr<const State> state) const {
        ThtsEnvContext default_ctx;
        return default_ctx;
    }
} 