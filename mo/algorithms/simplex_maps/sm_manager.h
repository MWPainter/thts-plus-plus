#pragma once

#include "mo/mo_thts_manager.h"

#include <limits>


namespace thts {
    
    // Forward declare
    class MoThtsEnv;
    struct Triangulation;

    // enum for how to which rules to use to split simplices
    enum ContextWeightOverwriteOption 
    {
        CONTEXT_WEIGHT_OVERWRITE_NONE = 0,                              // dont overwrite context weight
        CONTEXT_WEIGHT_OVERWRITE_UNIFORM_RANDOM_VERTEX = 1,   // overwrite context weight with a vertex sampled uniformly randomly from the simplex map at root node
        
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

        static const ContextWeightOverwriteOption context_weight_overwrite_option_default=CONTEXT_WEIGHT_OVERWRITE_NONE;

        int max_push_radius; // maximum number of hops to push value estimates to neighbours
        int max_neighbours_to_push_to; // maximum number of neighbours to push value estimates to from single node

        double min_simplex_radius_in_simplex_tree; // minimum radius (longest edge length) of simplices in simplex map, before stop splitting
        int max_depth_in_simplex_tree; // maximum depth of simplices in simplex map, before stop splitting
        int simplex_split_counter_threshold; // number of times in a row that vertexes must have different value estimates before subdividing

        bool use_approx_nearest_vertex; // whether to consider just containing simplex for vertex lookup (as opposed to searching 1-ring neighbourhood for actual closest)
        bool eventually_conforming_simplex_map; // split extra nodes each iteration to eventually enforce simplex mesh conformity
        bool always_allow_non_conforming_simplex_to_split; // even if above params' conditions are met

        ContextWeightOverwriteOption context_weight_overwrite_option; // if/how to overwrite context weight at root node

        SmThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            MoThtsManagerArgs(thts_env)
            max_push_radius(max_push_radius_default),
            max_neighbours_to_push_to(max_neighbours_to_push_to_default),
            min_simplex_radius_in_simplex_tree(min_simplex_radius_in_simplex_tree_default),
            max_depth_in_simplex_tree(max_depth_in_simplex_tree_default),
            simplex_split_counter_threshold(simplex_split_counter_threshold_default),
            use_approx_nearest_vertex(use_approx_nearest_vertex_default),
            eventually_conforming_simplex_map(eventually_conforming_simplex_map_default),
            always_allow_non_conforming_simplex_to_split(always_allow_non_conforming_simplex_to_split_default),
            context_weight_overwrite_option(context_weight_overwrite_option_default)
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

            ContextWeightOverwriteOption context_weight_overwrite_option; // if/how to overwrite context weight at root node

            SmThtsManager(const SmThtsManagerArgs& args) : 
                MoThtsManager(args),
                max_push_radius(args.max_push_radius),
                max_neighbours_to_push_to(args.max_neighbours_to_push_to),
                min_simplex_radius_in_simplex_tree(args.min_simplex_radius_in_simplex_tree),
                max_depth_in_simplex_tree(args.max_depth_in_simplex_tree),
                simplex_split_counter_threshold(args.simplex_split_counter_threshold),
                use_approx_nearest_vertex(args.use_approx_nearest_vertex),
                eventually_conforming_simplex_map(args.eventually_conforming_simplex_map),
                always_allow_non_conforming_simplex_to_split(args.always_allow_non_conforming_simplex_to_split),
                context_weight_overwrite_option(args.context_weight_overwrite_option)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmThtsManager() = default;
            
    };
}