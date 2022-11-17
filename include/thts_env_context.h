#pragma once

#include <string>
#include <unordered_map>

namespace thts {
    /**
     * A base context class for use in ThtsEnv objects (see below).
     * 
     * This class is basically a wrapper around a map that is used as a context. Its main purpose is to provide a base 
     * class that can be used by the ThtsEnv abstract class, and can be subclassed if a context more intricate than a 
     * map from strings to doubles is necessary.
     */
    class ThtsEnvContext {

        private:
            std::unordered_map<std::string, double> context;

        public:
            /**
             * Mark destructor virtual in case class is inherited from
             */
            virtual ~ThtsEnvContext() = default;

            /**
             * Gets a value from this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (double) stored in this context for the 'key'
             */
            virtual double get_value_for_key(const std::string& key);

            /**
             * Puts a value in this context for a given key string.
             * 
             * Args:
             *      key: A string used to lookup a value in this context
             * Returns:
             *      The value (double) stored in this context for the 'key'
             */
            virtual void put_value(const std::string& key, double val);
    };
}