#pragma once

#include "mo/mo_thts_manager.h"

#include "mo/simplex_map.h"

#include <limits>


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtThtsManagerArgs : public MoThtsManagerArgs {
        static const int num_backups_before_allowed_to_split_default = -1;

        static constexpr double simplex_node_l_inf_thresh_default = 0.05; 
        static const int simplex_node_split_visit_thresh_default = 10;
        static const int simplex_node_max_depth_default = std::numeric_limits<int>::max();

        Eigen::ArrayXd default_q_value;

        double simplex_node_l_inf_thresh;
        int simplex_node_split_visit_thresh;
        int simplex_node_max_depth;

        SmtThtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            MoThtsManagerArgs(thts_env),
            default_q_value(default_q_value),
            simplex_node_l_inf_thresh(simplex_node_l_inf_thresh_default),
            simplex_node_split_visit_thresh(simplex_node_split_visit_thresh_default),
            simplex_node_max_depth(simplex_node_max_depth_default)
        {
        }

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

            double simplex_node_l_inf_thresh;
            int simplex_node_split_visit_thresh;
            int simplex_node_max_depth;

            Triangulation triangulation;

            /**
             * Constructor.
             */    
            SmtThtsManager(const SmtThtsManagerArgs& args) : 
                MoThtsManager(args),
                default_q_value(args.default_q_value),
                simplex_node_l_inf_thresh(args.simplex_node_l_inf_thresh),
                simplex_node_split_visit_thresh(args.simplex_node_split_visit_thresh),
                simplex_node_max_depth(args.simplex_node_max_depth),
                triangulation(reward_dim)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtThtsManager() = default;
            
    };
}