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
            double solved_value;

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
             * If thts_manager->use_solved_labelling is true, then this set will only contain the children 
             * with the minimum solved_levels
             */
            std::vector<std::shared_ptr<const Action>> get_actions_to_consider(ThtsContext& ctx) const;

            /**
             * Returns the a label for "how solved" this node and the subtree under this node is.
             * A level of 0 means that the node is not solved
             * A level of 1 means that the node is solved to within a tolerance of tau
             * A level of i means that the node is solved to within a tolerance of tau / 2^(i-1)
             *
             * N.B. a node is solved when it's solved value is 0
             * If a node actually achieves a solved value of 0, then it's solved level is std::numeric_limits<int>::max()
             */
            int get_solved_level() const;

            /**
             * Returns the "solved value" of this node.
             * If a node is solved, then it's solved value is 0
             * If a node is not solved (i.e. it has never been seen), it's solved value is 1
             * 
             * The solved value of a decision node is the maximum solved value of all its children
             * Note that if an action has never been taken, then it's solved value is 1
             */
            double get_solved_value() const;

            /**
             * Update the solved value of this node.
             * The solved value is to be updated by the subclass.
             * I.e. update_solved_value() should update solved_value
             */
            void update_solved_value();

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
    };
}