#pragma once

#include "mo/mo_thts_manager.h"

#include <limits>


namespace thts {
    
    // Forward declare
    class MoThtsEnv;
    struct Triangulation;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtThtsManagerArgs : public MoThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        static const bool use_triangulation_default = false;

        static constexpr double simplex_node_l_inf_thresh_default = 0.05; 
        static const int simplex_node_split_visit_thresh_default = 10;
        static const int simplex_node_max_depth_default = std::numeric_limits<int>::max();

        Eigen::ArrayXd default_q_value;

        bool use_triangulation;

        double simplex_node_l_inf_thresh;
        int simplex_node_split_visit_thresh;
        int simplex_node_max_depth;

        SmtThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            MoThtsManagerArgs(thts_env),
            default_q_value(default_q_value),
            use_triangulation(use_triangulation_default),
            simplex_node_l_inf_thresh(simplex_node_l_inf_thresh_default),
            simplex_node_split_visit_thresh(simplex_node_split_visit_thresh_default),
            simplex_node_max_depth(simplex_node_max_depth_default)
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

            bool use_triangulation;

            double simplex_node_l_inf_thresh;
            int simplex_node_split_visit_thresh;
            int simplex_node_max_depth;

            std::shared_ptr<Triangulation> triangulation_ptr;

            /**
             * Constructor.
             */    
            SmtThtsManager(const SmtThtsManagerArgs& args) : 
                MoThtsManager(args),
                default_q_value(args.default_q_value),
                use_triangulation(args.use_triangulation),
                simplex_node_l_inf_thresh(args.simplex_node_l_inf_thresh),
                simplex_node_split_visit_thresh(args.simplex_node_split_visit_thresh),
                simplex_node_max_depth(args.simplex_node_max_depth),
                triangulation_ptr(std::make_shared<Triangulation>(args.use_triangulation ? reward_dim : 0))
            {
            };

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtThtsManager() = default;
            
    };
}