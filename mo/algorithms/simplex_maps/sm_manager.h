#pragma once

#include "mo/mo_thts_manager.h"

#include <limits>


namespace thts {
    
    // Forward declare
    class MoThtsEnv;
    struct Triangulation;

    // enum for how to which rules to use to split simplices
    enum SimplexMapSplittingOption 
    {
        SPLIT_ordered = 0,                  // split along edge with minimal ||w_1 - w_2||_inf minimised, ties broken by first edge found (will lead to the same order of splits and topology of graph each time)
        SPLIT_smallest_edge_randomly = 1,   // split along edge with minimal ||w_1 - w_2||_inf minimised, ties broken randomly
        SPLIT_random = 2,                   // split along a random edge (provided ||w_1 - w_2||_inf < threshold)
        SPLIT_value_diff = 3,               // split along the edge with maximal value of ||val_1 - val_2||_2
        SPLIT_triangulation = 4,            // split simplices using a triangulation (computed in python), rather than bin tree
    };

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmThtsManagerArgs : public MoThtsManagerArgs {
        static const int max_push_radius_default=1;
        static const int max_neighbours_to_push_to_default=-1;

        static const double min_simplex_radius_in_simplex_tree_default=0.01;
        static const int max_depth_in_simplex_tree_default=std::numeric_limits<int>::max();
        static const int simplex_split_counter_threshold_default=10;

        static const bool use_approx_nearest_vertex_default=false;
        static const bool eventually_conforming_simplex_map_default=true;
        static const bool always_allow_non_conforming_simplex_to_split_default=true;

        int max_push_radius; // maximum number of hops to push value estimates to neighbours
        int max_neighbours_to_push_to; // maximum number of neighbours to push value estimates to from single node

        double min_simplex_radius_in_simplex_tree; // minimum radius (longest edge length) of simplices in simplex map, before stop splitting
        int max_depth_in_simplex_tree; // maximum depth of simplices in simplex map, before stop splitting
        int simplex_split_counter_threshold; // number of times in a row that vertexes must have different value estimates before subdividing

        bool use_approx_nearest_vertex; // whether to consider just containing simplex for vertex lookup (as opposed to searching 1-ring neighbourhood for actual closest)
        bool eventually_conforming_simplex_map; // split extra nodes each iteration to eventually enforce simplex mesh conformity
        bool always_allow_non_conforming_simplex_to_split; // even if above params' conditions are met




        SmThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            MoThtsManagerArgs(thts_env)
            max_push_radius(max_push_radius_default),
            max_neighbours_to_push_to(max_neighbours_to_push_to_default),
            min_simplex_radius_in_simplex_tree(min_simplex_radius_in_simplex_tree_default),
            max_depth_in_simplex_tree(max_depth_in_simplex_tree_default),
            simplex_split_counter_threshold(simplex_split_counter_threshold_default),
            use_approx_nearest_vertex(use_approx_nearest_vertex_default),
            eventually_conforming_simplex_map(eventually_conforming_simplex_map_default),
            always_allow_non_conforming_simplex_to_split(always_allow_non_conforming_simplex_to_split_default)
        {
        }

        virtual ~SmThtsManagerArgs() = default;
    };

    class SmThtsManager : public MoThtsManager {
        public:

            int max_push_radius; // maximum number of hops to push value estimates to neighbours
            int max_neighbours_to_push_to; // maximum number of neighbours to push value estimates to from single node

            double min_simplex_radius_in_simplex_tree; // minimum radius (longest edge length) of simplices in simplex map, before stop splitting
            int max_depth_in_simplex_tree; // maximum depth of simplices in simplex map, before stop splitting
            int simplex_split_counter_threshold; // number of times in a row that vertexes must have different value estimates before subdividing

            bool use_approx_nearest_vertex; // whether to consider just containing simplex for vertex lookup (as opposed to searching 1-ring neighbourhood for actual closest)
            bool eventually_conforming_simplex_map; // split extra nodes each iteration to eventually enforce simplex mesh conformity
            bool always_allow_non_conforming_simplex_to_split; // even if above params' conditions are met

            SmThtsManager(const SmThtsManagerArgs& args) : 
                MoThtsManager(args),
                max_push_radius(args.max_push_radius),
                max_neighbours_to_push_to(args.max_neighbours_to_push_to),
                min_simplex_radius_in_simplex_tree(args.min_simplex_radius_in_simplex_tree),
                max_depth_in_simplex_tree(args.max_depth_in_simplex_tree),
                simplex_split_counter_threshold(args.simplex_split_counter_threshold),
                use_approx_nearest_vertex(args.use_approx_nearest_vertex),
                eventually_conforming_simplex_map(args.eventually_conforming_simplex_map),
                always_allow_non_conforming_simplex_to_split(args.always_allow_non_conforming_simplex_to_split)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmThtsManager() = default;
            
    };
}