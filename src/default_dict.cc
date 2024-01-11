#include "default_dict.h"


namespace thts::helper {
    using namespace std;

    /**
     * default dict
    */
    template<typename Key, typename T> 
    unordered_map_with_default<Key,T>::unordered_map_with_default(T default_value) : 
        default_value(default_value),
        map()
    {
    }

    template <typename Key, typename T>
    T& unordered_map_with_default<Key,T>::operator[](const Key& key) 
    {   
        if (!map.contains(key)) {
            map[key] = default_value;
        }
        return map[key];
    } 
}