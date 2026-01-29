#include "main_mo/configs/config_map.h"

// Helper to check if a config value is of type std::vector<int>
bool config_value_is_int_vector(const ConfigMap& config, const std::string& key)
{
    if (!config.contains(key)) {
        return false;
    }
    return std::holds_alternative<std::vector<int>>(config.at(key));
}