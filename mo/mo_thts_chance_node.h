#pragma once

#include "thts_chance_node.h"
#include "thts_manager.h"
#include "thts_types.h"
#include "mo/mo_thts_decision_node.h"
#include "mo/mo_thts_manager.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    
    // forward declare
    class MoThtsDNode;
    
    /**
     * An abstract base class for multi objective Chance Nodes.
     */
    class MoThtsCNode : public ThtsCNode {
        // Allow ThtsDNode access to private members
        friend MoThtsDNode;

        protected:
            Vec vector_visit_count;
            int local_backups;
            int total_cnode_backups_in_subtree;
            int total_dnode_backups_in_subtree;
            int solved_labelling;

        public: 
            /**
             * Default constructor.
             * 
             * Initialises the attributes of the class.
             */
            MoThtsCNode(
                std::shared_ptr<MoThtsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MoThtsDNode> parent=nullptr);

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MoThtsCNode() = default;

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
            int get_solved_labelling() const;

            /**
             * Get a confidence interval to estimate how "solved" this node is.
             *
             * Part 0: What we're trying to compute a bound for
             *    Roughly we have a true distribution of outcomes, p, and we want to compute some confidence intervals
             *    If we have some interval (range) for each outcome, r(x), then our expected range is:
             *        r = sum_x p(x)r(x)
             *
             *    However, we don't have the true distribtion p, and we don't know if we have the seen all outcomes yet
             *    So we will split this into two parts:
             *          r = sum_x_seen p(x)r(x) + sum_x_missing p(x)r(x) = r_seen + r_missing
             * 
             * Part 1: DKW inequality, implies bounds on the empirical distribution
             *    With, empirical distribtuion q, true distribution p, outcome x, and prob > 1-delta:
             *        p(x) <= q(x) + 2 sqrt(log(2/delta) / (2 * n)) 
             *        Let q'(x_) = q(x) + 2 sqrt(log(2/delta) / (2 * n)) 
             *
             * Part 2: Range of confidence interval at this node:
             *        Let r(x) be the range of the confidence interval for outcome x (from child)
             *        Then r_seen, the range at this node (assuming we have seen all outcomes), can be bounded by:
             *            r_seen = sum_x p(x)r(x) <= sum_x q'(x)r(x)
             *
             * Part 3: Missing mass
             *  We also need to account for the mass that may be missing from the empirical distribution
             *        (I.e. we may have not seen all possible outcomes yet)
             *        We will use a Good Turing estimate to estimate the missing mass
             *        Let M be the total missing mass, c be the number of outcomes seen exactly once, and n to total number of observations
             *            M = sum_x_missing p(x)
             *        The Good Turing estimate is then:
             *            M' = c/n
             *        With probability > 1-delta, we have:
             *            M <= M' + sqrt(log(1/delta) / n)
             *        Then r_missing, the range at this node (assuming we have not seen all outcomes), can be bounded by:
             *            r_missing = sum_x_missing p(x)r(x) <=  r_max * (M' + sqrt(log(1/delta) / n))
             *
             *        Then the final confidence interval range is:
             *            r = r_seen + r_missing 
             *              = sum_x_seen [r(x) * (q(x) + 2 sqrt(log(2/delta) / (2 * n))) ] 
             *                  + r_max * (M' + sqrt(log(1/delta) / n))
             */
            double get_solved_labelling_confidence_interval_range() const;

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
             * Updates the Sinformation in this node in the backup phase of the thts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking the action 
             *          in this node.
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

            double get_num_visits(MoThtsContext& ctx) const;
            double get_scalar_num_visits() const;
            Vec get_vector_num_visits() const;
            virtual void visit_itfc(ThtsContext& ctx) override;
            
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