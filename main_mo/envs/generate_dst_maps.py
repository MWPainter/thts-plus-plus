#!/usr/bin/env python3
"""
Implements a generalised form of the Deep Sea Treasure MORL benchmark problem as originally described in Vamplew et al (2011).
Instances of the benchmark can be created varying in terms of the size and structure of the state space, the level of stochasticity in
both the transitions and the rewards, and the shape of the front defined by the rewards.
To create, call the script passing in the following arguments:
- the width of the environment (number of columns) - a positive int
- the starting depth of the sea-bed - a positive int (may have some effect on different exploration strategies)
- the minimum depth difference between neighbouring columns - int >= 0
- the maximum depth difference between neighbouring columns - int >= 0
- transition noise - the probability that an action will be taken at random rather than as specified by the agent - double between 0 and 1 inclusive (not used for map generation)
- reward noise - the standard variation of Gaussian noise to be added to reward values on each time-step (not used for map generation)
- front shape - linear, concave, convex or mixed (use the constants: LINEAR=0, CONCAVE=1, CONVEX=2, MIXED=3)
- seed - random seed for reproducibility
Written by Peter Vamplew August 2017 (Java version)
Converted to Python for map generation
"""

import sys
import random
import argparse
import math

# Constants for front shapes
LINEAR = 0
CONCAVE = 1
CONVEX = 2
MIXED = 3

MIN_TREASURE = 1
MAX_TREASURE = 1000


def construct_environment(r, width, min_depth, min_vertical_step, max_vertical_step, seed, use_x_plus_y_max=True, scale=1.0, curvature_k=1.0):
    """
    Sets up the properties of the environment based on the provided parameters.
    Returns a list of treasure positions and rewards in the format [[[x, y], reward], ...]
    
    Args:
        use_x_plus_y_max: If True, max reward equals max(x+y) from treasure locations. 
                         If False, max reward is 1000.
        scale: Multiplicative scale to apply to all treasure rewards.
    """
    # Set up the structure of the environment
    num_cols = width
    depths = [0] * num_cols
    steps = [0] * num_cols
    depths[0] = min_depth
    steps[0] = min_depth
    step_range = max_vertical_step - min_vertical_step + 1
    
    for col in range(1, num_cols):
        min_vertical_step_this_col = min_vertical_step
        step_range_this_col = step_range
        if col == 1:
            min_vertical_step_this_col = 1
            step_range_this_col = step_range - 1
        depths[col] = depths[col-1] + r.randint(0, step_range_this_col - 1) + min_vertical_step_this_col
        steps[col] = col + depths[col]
    
    # Calculate treasure rewards
    treasure = set_treasure(num_cols, steps, use_x_plus_y_max, depths, scale, curvature_k)
    
    # Generate output in the required format: [[[x, y], reward], ...]
    result = []
    for col in range(num_cols):
        result.append([[col, depths[col]], float(treasure[col])])
    
    return result


def set_treasure(num_cols, steps, use_x_plus_y_max, depths, scale, curvature_k):
    """
    Sets the treasure reward values based on the front shape.
    Returns an array of treasure values for each column.
    
    Args:
        curvature_k: Exponent applied to the adjustment factor to control front curvature.
                           Higher values make convex/concave shapes more pronounced.
    """
    treasure = [0] * num_cols
    
    # Determine max treasure value
    if use_x_plus_y_max:
        # Calculate max(x+y) from treasure locations
        max_treasure_val = max(col + depths[col] for col in range(num_cols))
    else:
        max_treasure_val = MAX_TREASURE
    
    treasure[0] = MIN_TREASURE
    treasure[num_cols-1] = max_treasure_val
    
    steps_range = steps[num_cols-1] - steps[0]
    treasure_range = max_treasure_val - MIN_TREASURE
    
    for col in range(1, num_cols-1):
        # treasure_x = ratio of num steps to reach this treasure to max treasure value
        # treasure_y = ratio of treasure between min an max values
        # (treasure_x,treasure_y) ranges from (0,0) to (1,1) in a concave shape to give the convex front
        treasure_x = (steps[col] - steps[0]) / float(steps_range) if steps_range > 0 else 0
        treasure_y = (1.0 - math.exp(-curvature_k * treasure_x)) / (1.0 - math.exp(-curvature_k))
        treasure[col] = treasure_y * treasure_range + MIN_TREASURE
    
    # Apply multiplicative scale to all treasures
    treasure = [t * scale for t in treasure]
    
    return treasure


def pretty_print_map(result):
    """
    Pretty prints the map in the format matching custom_deep_sea_treasure_maps.py
    """
    lines = ["    ["]
    for i, item in enumerate(result):
        x, y = item[0]
        reward = item[1]
        if i == len(result) - 1:
            lines.append(f"        [[{x}, {y}], {reward:.6f}]")
        else:
            lines.append(f"        [[{x}, {y}], {reward:.6f}],")
    lines.append("    ],")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a Deep Sea Treasure map configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python custom_deep_sea_treasure_generate_map.py --width 10
  python custom_deep_sea_treasure_generate_map.py --width 10 --front-shape CONVEX --seed 471
  python custom_deep_sea_treasure_generate_map.py --widths 10 15 20 --min-depth 1 --max-vertical-step 3
        """
    )
    
    parser.add_argument('--width', type=int, default=10,
                        help='Width of the environment (number of columns) - positive int (default: 10)')
    parser.add_argument('--widths', type=int, nargs='+',
                        help='List of widths to generate (overrides --width if provided)')
    parser.add_argument('--min-depth', type=int, default=1,
                        help='Starting depth of the sea-bed - positive int (default: 1)')
    parser.add_argument('--min-vertical-step', type=int, default=0,
                        help='Minimum depth difference between neighbouring columns - int >= 0 (default: 0)')
    parser.add_argument('--max-vertical-step', type=int, default=3,
                        help='Maximum depth difference between neighbouring columns - int >= 0 (default: 3)')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='Multiplicative scale to apply to all treasure rewards (default: 2.0)')
    parser.add_argument('--seed', type=int, default=471,
                        help='Random seed for reproducibility (default: 471)')
    parser.add_argument('--no-use-x-plus-y-max', dest='use_x_plus_y_max', action='store_false',
                        default=True,
                        help='Use max reward of 1000 instead of max(x+y). Default: max reward equals max(x+y)')
    parser.add_argument('--curvature-k', type=float, default=1.0,
                        help='Constant controlling front curvature. Higher values make convex/concave shapes more pronounced (default: 1.0)')
    
    args = parser.parse_args()
    
    # Determine which widths to generate
    if args.widths:
        widths = args.widths
    else:
        widths = [args.width]
    
    # Validate arguments
    for width in widths:
        if width <= 0:
            print(f"Error: width must be a positive integer, got {width}", file=sys.stderr)
            sys.exit(1)
    if args.min_depth <= 0:
        print("Error: min_depth must be a positive integer", file=sys.stderr)
        sys.exit(1)
    if args.min_vertical_step < 0:
        print("Error: min_vertical_step must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.max_vertical_step < args.min_vertical_step:
        print("Error: max_vertical_step must be >= min_vertical_step", file=sys.stderr)
        sys.exit(1)
    
    # Generate maps for each width
    try:
        all_results = []
        r = random.Random(args.seed)
        for width in widths:
            result = construct_environment(
                r,
                width,
                args.min_depth,
                args.min_vertical_step,
                args.max_vertical_step,
                args.seed,
                args.use_x_plus_y_max,
                args.scale,
                args.curvature_k
            )
            all_results.append((width, result))
        
        # Output each map in pretty format
        for width, result in all_results:
            print(f"    {width}:")
            print(pretty_print_map(result))
            if width != all_results[-1][0]:  # Add blank line between maps if multiple
                print()
        
        # Print dictionary mapping width to max(x+y)
        width_to_max_xy = {}
        for width, result in all_results:
            max_xy = max(item[0][0] + item[0][1] for item in result)
            width_to_max_xy[width] = max_xy
        
        print(f"\n# Width to max(x+y) mapping:")
        print("{")
        for width, max_xy in width_to_max_xy.items():
            print(f"    {width}: {max_xy},")
        print("}")
        
    except Exception as e:
        print(f"Error generating map: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

