#pragma once

#include "thts_decision_node.h"
#include "thts_manager.h"
#include "mo/mo_thts_chance_node.h"
#include "mo/mo_thts_manager.h"

#include "mo/data_structures/convex_hull.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace thts {
    
    // forward declare
    class MoThtsCNode;
    class ThtsLogger;
    class MoThtsPool;
    class MoThtsContext;

    /**
     * An abstract base class for a multi objective Decision Node.
     * 
     * Note that this class has a scalar 'heuristic_value' and a 'mo_heuristic_value'. Decided it's not worth 
     * seperating it out into DNode (without 'heuristic_value') and ThtsDNode for scalar objectives for now. If 
     * multi-objective methods become more mainstream then it might be worth optimising the library for memory for 
     * multi-objective algorithms, but this will be very marginal anyway.
     * 
     * Member variables:
     *      mo_heuristic_value:
     *          The (multi objective) heuristic value of this decision node
     */
    class MoThtsDNode : public ThtsDNode {
        // Allow ThtsCNode, Logger and Pool access to private members
        friend MoThtsCNode;
        friend ThtsLogger;
        friend MoThtsPool;
        friend MoThtsContext;

        protected:
            Vec mo_heuristic_value;
            Vec vector_visit_count;
            int local_backups;
            int total_cnode_backups_in_subtree;
            int total_dnode_backups_in_subtree;
            int solved_labelling;
            double solved_labelling_confidence_interval_range;

        public: 
            /**
             * Constructor.
             * 
             * Initialises the attributes of the class.
             */
            MoThtsDNode(
                std::shared_ptr<MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MoThtsCNode> parent=nullptr,
                bool eval_mo_heuristic=true); 

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MoThtsDNode() = default;

            /**
             * Returns the set of actions to consider for selection.
             *
             * If thts_manager->use_solved_labelling is true, then this set will only contain the children minimum solved_labellings
             */
            std::vector<std::shared_ptr<const Action>> get_actions_to_consider() const;

            /**
             * Returns the a label for "how solved" this node is.
             * Let delta be the the size of a confidence interval at this node
             * If tau is the threshold acceptible for considering this node "solved"
             * This function return the value: min_i s.t. delta > tau / 2^i
             * 
             * I.e. returning a value of 0 means that this node is not solved
             * Returning a value of 1 means that this node is solved to within a tolerance of tau
             * Further values indicate node is solved to further and further tolerances
             */
            int get_local_solved_labelling() const;

            /**
             * Returns the a label for "how solved" the subtree under this node is.
             * That is, it returns the minimum of the local_solved_labelling and the solved_labelling all children
             * I.e. a decision node is only solved if it is confident in its decision and all its children are solved
             */
            int get_solved_labelling() const;

            /**
             * Get a local confidence interval to estimate how "solved" this node is.
             * If this node is not solved, return the maximum range. 
             * If node is solved, then return the confidence interval range cached.
             *
             * Local is the version to use internally in the node
             */
            double get_solved_labelling_confidence_interval_range() const;
        private:
            double get_local_solved_labelling_confidence_interval_range() const;
        public:

            /**
             * Update the solved labelling of this node.
             * The confidence interval range is to be updated by the subclass.
             * I.e. update_solved_labelling_confidence_interval_range() should update solved_labelling_confidence_interval_range
             */
            void update_solved_labelling();
            virtual void update_solved_labelling_confidence_interval_range() = 0;

            /**
             * OVerride final the old backup fn (throws error if try to call)
            */
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsContext& ctx) override final;

            /**
             * Override of thts backup function for multi objective.
             * 
             * Updates the information in this node in the backup phase of the thts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
             */
            virtual void backup_itfc(
                const std::vector<Eigen::ArrayXd>& trial_rewards_before_node, 
                const std::vector<Eigen::ArrayXd>& trial_rewards_after_node, 
                const Eigen::ArrayXd trial_cumulative_return_after_node, 
                const Eigen::ArrayXd trial_cumulative_return,
                ThtsContext& ctx) = 0;
            

            /**
             * THTS interface
             */
            virtual void visit_itfc(ThtsContext& ctx) override;

            /**
             * Get the number of visits, possibly using contextual visit counts
             */
            double get_num_visits(MoThtsContext& ctx) const;
            double get_scalar_num_visits() const;
            Vec get_vector_num_visits() const;

            /**
             * Get an (approximate) convex hull from this node
             */
            virtual ConvexHull get_convex_hull() const;
            
            /**
             * Logging to keep track of total number of backups
             * Want to use this to compare the efficiencies of different data structures
             */
            void increment_and_update_backup_count();
            int get_total_backups_in_subtree();
            int get_cnode_backups_in_subtree();
            int get_dnode_backups_in_subtree();
    };
}