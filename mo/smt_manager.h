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
    struct SmtThtsManagerArgs : public MoThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        static const SimplexMapSplittingOption simplex_map_splitting_option_default = SPLIT_value_diff;

        static constexpr double simplex_node_l_inf_thresh_default = 0.05; 
        static const int simplex_node_split_visit_thresh_default = 10;
        static const int simplex_node_max_depth_default = std::numeric_limits<int>::max();

        static const bool backup_all_vertices_of_simplex_default = false;

        Eigen::ArrayXd default_q_value;

        SimplexMapSplittingOption simplex_map_splitting_option;

        double simplex_node_l_inf_thresh;
        int simplex_node_split_visit_thresh;
        int simplex_node_max_depth;

        bool backup_all_vertices_of_simplex;

        SmtThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            MoThtsManagerArgs(thts_env),
            default_q_value(default_q_value),
            simplex_map_splitting_option(simplex_map_splitting_option_default),
            simplex_node_l_inf_thresh(simplex_node_l_inf_thresh_default),
            simplex_node_split_visit_thresh(simplex_node_split_visit_thresh_default),
            simplex_node_max_depth(simplex_node_max_depth_default),
            backup_all_vertices_of_simplex(backup_all_vertices_of_simplex_default)
        {
        };

        virtual ~SmtThtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      num_backups_before_allowed_to_split:
     *          The number of backups that have to be performed at a CZ_Ball before it is allowed to 'split' and 
     *          create child balls.
     */
    class SmtThtsManager : public MoThtsManager {
        public:
            Eigen::ArrayXd default_q_value;

            SimplexMapSplittingOption simplex_map_splitting_option;

            double simplex_node_l_inf_thresh;
            int simplex_node_split_visit_thresh;
            int simplex_node_max_depth;

            bool backup_all_vertices_of_simplex;

            std::shared_ptr<Triangulation> triangulation_ptr;

            /**
             * Constructor.
             */    
            SmtThtsManager(const SmtThtsManagerArgs& args) : 
                MoThtsManager(args),
                default_q_value(args.default_q_value),
                simplex_map_splitting_option(args.simplex_map_splitting_option),
                simplex_node_l_inf_thresh(args.simplex_node_l_inf_thresh),
                simplex_node_split_visit_thresh(args.simplex_node_split_visit_thresh),
                simplex_node_max_depth(args.simplex_node_max_depth),
                backup_all_vertices_of_simplex(args.backup_all_vertices_of_simplex),
                triangulation_ptr(std::make_shared<Triangulation>(
                    (args.simplex_map_splitting_option == SPLIT_triangulation) ? reward_dim : 0))
            {
            };

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtThtsManager() = default;
            
    };
}