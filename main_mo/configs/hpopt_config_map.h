#pragma once

#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>


// ---------------------------------------------------------------------------------------------------------------------
// Configs for hpopt will be a vector of ConfigMap types. Typedefs + helper functions for actual configs:
// ---------------------------------------------------------------------------------------------------------------------

// Type aliases
using ConfigValueRange = std::variant<std::string, bool, int, double, std::pair<int,int>, std::pair<double,double>>;
using HpoptConfigMap   = std::unordered_map<std::string, ConfigValueRange>;

// Templated Helper to read value from config map (with numeric type casting support)
template<typename T>
T get_config_value(const HpoptConfigMap& config, const std::string& key)
{
    if (!config.contains(key)) {
        std::stringstream err_msg;
        err_msg << "Expecting to find key (" << key << ") in ConfigMap, but couldn't.";
        throw std::runtime_error(err_msg.str());
    }

    const ConfigValueRange& value = config.at(key);

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

    // Handle pair conversions (pair<int,int> -> pair<double,double>)
    if constexpr (std::is_same_v<T, std::pair<double,double>>) {
        if (auto* ptr = std::get_if<std::pair<int,int>>(&value)) {
            return std::pair<double,double>(static_cast<double>(ptr->first), static_cast<double>(ptr->second));
        }
    } 

    // No valid conversion found - throw with helpful message
    std::stringstream err_msg;
    err_msg << "Type mismatch for key (" << key << "): cannot convert stored type to requested type.";
    throw std::runtime_error(err_msg.str());
}