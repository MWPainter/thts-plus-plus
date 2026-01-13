#!/usr/bin/env python3
"""
Script to generate custom_deep_sea_treasure_maps.cpp from custom_deep_sea_treasure_maps.py
Reads the Python file and outputs the corresponding C++ file with the correct format.
"""

import sys
import os

def load_python_data(py_file_path):
    """Load WIDTH_TO_MAX_XY and MAPS from the Python file by executing it."""
    # Read the Python file
    with open(py_file_path, 'r') as f:
        py_content = f.read()
    
    # Create a namespace to execute the Python file
    namespace = {}
    exec(py_content, namespace)
    
    width_to_max_xy = namespace.get('WIDTH_TO_MAX_XY', {})
    maps = namespace.get('MAPS', {})
    
    return width_to_max_xy, maps

def format_float(value):
    """Format a float value for C++ output, removing unnecessary trailing zeros."""
    if isinstance(value, int):
        return str(float(value))
    # Format to handle precision appropriately
    formatted = f"{value:.6f}".rstrip('0').rstrip('.')
    if formatted == '':
        return '0.0'
    if '.' not in formatted:
        formatted += '.0'
    return formatted

def generate_cpp_file(width_to_max_xy, maps, output_path):
    """Generate the C++ file from the Python data."""
    
    lines = []
    lines.append('#include "custom_deep_sea_treasure_maps.h"')
    lines.append('')
    lines.append('using namespace std;')
    lines.append('')
    lines.append('namespace thts {')
    lines.append('')
    lines.append('// WIDTH_TO_MAX_XY mapping')
    lines.append('const unordered_map<int, int> WIDTH_TO_MAX_XY = {')
    
    # Sort keys for consistent output
    sorted_widths = sorted(width_to_max_xy.keys())
    for i, width in enumerate(sorted_widths):
        max_xy = width_to_max_xy[width]
        if i < len(sorted_widths) - 1:
            lines.append(f'    {{ {width:3}, {max_xy}}},')
        else:
            lines.append(f'    {{ {width:3}, {max_xy}}},')
    
    lines.append('};')
    lines.append('')
    lines.append('// MAPS data')
    lines.append('const unordered_map<int, TreasureMap> MAPS = {')
    
    # Sort map IDs for consistent output
    sorted_map_ids = sorted(maps.keys())
    for map_id in sorted_map_ids:
        treasure_map = maps[map_id]
        lines.append(f'    {{{map_id}, TreasureMap{{')
        
        # Add each treasure entry
        for entry in treasure_map:
            x, y = entry[0]
            reward = entry[1]
            formatted_reward = format_float(reward)
            lines.append(f'        TreasureEntry({x}, {y}, {formatted_reward}),')
        
        # Remove the trailing comma from the last entry
        if treasure_map:
            # Replace the last line's comma with nothing for the last entry
            last_line_idx = len(lines) - 1
            lines[last_line_idx] = lines[last_line_idx].rstrip(',') + ''
        
        lines.append('    }},')
    
    lines.append('};')
    lines.append('')
    lines.append('const TreasureMap* get_map(int map_id) {')
    lines.append('    auto it = MAPS.find(map_id);')
    lines.append('    if (it == MAPS.end()) {')
    lines.append('        return nullptr;')
    lines.append('    }')
    lines.append('    return &(it->second);')
    lines.append('}')
    lines.append('')
    lines.append('int get_max_xy_for_width(int width) {')
    lines.append('    auto it = WIDTH_TO_MAX_XY.find(width);')
    lines.append('    if (it == WIDTH_TO_MAX_XY.end()) {')
    lines.append('        return -1;')
    lines.append('    }')
    lines.append('    return it->second;')
    lines.append('}')
    lines.append('')
    lines.append('}')
    lines.append('')
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def main():
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_file = os.path.join(script_dir, 'custom_deep_sea_treasure_maps.py')
    cpp_file = os.path.join(script_dir, 'custom_deep_sea_treasure_maps.cpp')
    
    # Check if Python file exists
    if not os.path.exists(py_file):
        print(f"Error: Python file not found: {py_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load data from Python file
        print(f"Loading data from {py_file}...")
        width_to_max_xy, maps = load_python_data(py_file)
        print(f"Loaded {len(width_to_max_xy)} width mappings and {len(maps)} maps")
        
        # Generate C++ file
        print(f"Generating C++ file: {cpp_file}...")
        generate_cpp_file(width_to_max_xy, maps, cpp_file)
        print(f"Successfully generated {cpp_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

