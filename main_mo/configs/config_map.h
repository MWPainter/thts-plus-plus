#pragma once

#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
#include <stdexcept>
#include <type_traits>


// ---------------------------------------------------------------------------------------------------------------------
// Configs will be a vector of ConfigMap types. Typedefs + helper functions for actual configs:
// ---------------------------------------------------------------------------------------------------------------------

// Type aliases
using ConfigValue = std::variant<std::string, bool, int, double, std::vector<int>>;
using ConfigMap   = std::unordered_map<std::string, ConfigValue>;

// Templated Helper to read value from config map (with numeric type casting support)
template<typename T>
T get_config_value(const ConfigMap& config, const std::string& key)
{
    if (!config.contains(key)) {
        std::stringstream err_msg;
        err_msg << "Expecting to find key (" << key << ") in ConfigMap, but couldn't.";
        throw std::runtime_error(err_msg.str());
    }

    const ConfigValue& value = config.at(key);

    // Try exact type match first
    if (auto* ptr = std::get_if<T>(&value)) {
        return *ptr;
    }

    // Handle numeric conversions (int -> double)
    if constexpr (std::is_same_v<T, double>) {
        if (auto* ptr = std::get_if<int>(&value)) {
            return static_cast<double>(*ptr);
        }
    }

    // No valid conversion found - throw with helpful message
    std::stringstream err_msg;
    err_msg << "Type mismatch for key (" << key << "): cannot convert stored type to requested type.";
    throw std::runtime_error(err_msg.str());
}

// Helper to check if a config value is of type std::vector<int>
bool config_value_is_int_vector(const ConfigMap& config, const std::string& key);