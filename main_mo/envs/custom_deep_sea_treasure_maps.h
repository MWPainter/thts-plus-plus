#pragma once

#include <vector>
#include <unordered_map>
#include <array>

namespace thts {
    /**
     * Treasure entry: position [x, y] and value
     */
    struct TreasureEntry {
        std::array<int, 2> position;  // [x, y]
        double value;
        
        TreasureEntry(int x, int y, double val) : position({x, y}), value(val) {}
    };

    /**
     * Map data: list of treasure entries
     */
    typedef std::vector<TreasureEntry> TreasureMap;

    /**
     * Width to max(x+y) mapping
     * Defined in the .cpp file
     */
    extern const std::unordered_map<int, int> WIDTH_TO_MAX_XY;

    /**
     * All available maps
     * Defined in the .cpp file
     */
    extern const std::unordered_map<int, TreasureMap> MAPS;

    /**
     * Get a specific map by ID
     * Returns nullptr if map doesn't exist
     */
    const TreasureMap* get_map(int map_id);

    /**
     * Get width to max_xy mapping
     * Returns -1 if width doesn't exist
     */
    int get_max_xy_for_width(int width);
}

