#include "thts_context.h"

using namespace std;

namespace thts { 

    /**
     * Returns if context contains a value for a given key string. (in the context map)
     */
    bool ThtsContext::context_map_contains(const string& key) const {
        return context.contains(key);
    }

    /**
     * (Const version) Returns if context contains a value for a given key string. (in the context map)
     */
    bool ThtsContext::const_context_map_contains(const string& key) const {
        return context_const.contains(key);
    }

    /**
     * Implementation of 'get_value_for_key'
     * 
     * Just try to access the private context map. If there is an error in the access, we want it to be thrown anyway.
     */
    shared_ptr<void> ThtsContext::get_value_raw(const string& key) const {
        return context.at(key);
    }
    /**
     * (Const version) Implementation of 'get_value_for_key'
     * 
     * Just try to access the private context map. If there is an error in the access, we want it to be thrown anyway.
     */
    shared_ptr<const void> ThtsContext::get_value_raw_const(const string& key) const {
        return context_const.at(key);
    }

    /**
     * Implementation of 'put_value'
     */
    void ThtsContext::put_value_raw(const string& key, shared_ptr<void> val) {
        context[key] = val;
    }

    /**
     * (Const version) Implementation of 'put_value' for pointers to const types
     */
    void ThtsContext::put_value_raw_const(const string& key, shared_ptr<const void> val) {
        context_const[key] = val;
    }

    /**
     * Remove object from context
    */
   void ThtsContext::erase(const string& key) {
        context.erase(key);
   }

    /**
     * (Const version) Remove object from context
    */
   void ThtsContext::erase_const(const string& key) {
        context_const.erase(key);
   }
}