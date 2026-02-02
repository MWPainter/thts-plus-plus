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
            double solved_value;

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
             * The solved value of a chance node the expected value of it's children's solved values
             * But this computation takes into account the "missing mass" of unseen outcomes and 
             * the uncertainty in the empirical distribution of the children's solved values
             * Maths for that in the update_solved_value function docstring below
             */
            double get_solved_value() const;

            /**
             * Update the solved value of this node.
             *
             * Let s(x) be the solved value of child x
             * Let q(x) be the empirical probability of outcome x
             * Let p(x) be the true probability of outcome x
             * 
             * What we really want to compute here is the expected value of the children's solved values,
             * that is:
             *        E[s] = sum_x p(x)s(x)
             *
             *  Because we don't know the true distribution p, we need to use the empirical distribution q, and 
             *  we will compute an upper bound on E[s] to use as our solved value instead
             *
             *  Let delta be the probability that our upper bound is violated.
             *  That is, we will compute E_bound such that Pr(E[s] > E_bound) <= 2*delta.
             *
             *  We split the expected value into two parts:
             *      E[s] = sum_(x_seen) p(x)s(x) + sum_(x_missing) p(x)s(x)
             *          = E[s]_seen + E[s]_missing
             *
             *  From the DKW inequality, we can get bounds on the empirical distribution (knowing that ours is 
             *  catagorical), with probability > 1-delta:
             *      p(x) <= q(x) + 2 sqrt(log(2/delta) / (2 * n))
             *      Let q'(x) = q(x) + 2 sqrt(log(2/delta) / (2 * n))
             *
             *  Then E[s]_seen can be bounded by:
             *      E[s]_seen = sum_(x_seen) p(x)s(x)
             *          <= sum_(x_seen) q'(x)s(x)
             *
             *  For the missing mass, we can use the Good Turing estimate. 
             *  Let M be the total missing mass, 
             *  Let c be the number of outcomes seen exactly once, 
             *  Let n be the total number of observations
             *  The Good Turing estimate is then:
             *      M' = c/n
             *
             *  Given our solved values are within the range [0,1], and 1 is used for unsolved nodes, we can assign a 
             *  value of 1 to the missing mass. I.e. for an unseen outcome x, we will use s(x) = 1
             *
             *  We can now bound E[s]_missing by:
             *      E[s]_missing = sum_(x_missing) p(x)s(x)
             *          <= sum_(x_missing) p(x)
             *          = M
             *          <= M' + sqrt(log(1/delta) / n)
             *
             *  Finally, by union bound (over the probability of failure of thw two bounds), 
             *  we have with probability > 1-2*delta that:
             *      E[s] <= E[s]_seen + E[s]_missing 
             *          = E_bound
             *          := sum_x_seen [s(x) * (q(x) + 2 sqrt(log(2/delta) / (2 * n))) ] 
             *                  + M' + sqrt(log(1/delta) / n)
             *
             * As a final note, we use mo_thts_manager.solver_labelling_delta_fail_probability as the value of 2*delta
             */
            void update_solved_value() const;

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