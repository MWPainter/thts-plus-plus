#pragma once

#include <Eigen/Dense>

#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "mo/data_structures/convex_hull.h"


namespace thts {
    // Frwd
    class CzBallList;

    /**
     * CzBall implementation
    */
    class CzBall {
        friend CzBallList;

        protected:
            double _radius;
            Eigen::ArrayXd _center;

            mutable std::mutex stats_lock;
            int num_backups;
            Eigen::ArrayXd avg_return_or_value;

        public:
            CzBall(const double radius, const Eigen::ArrayXd& center);

            int dim() const;

            bool point_in_domain(const Eigen::ArrayXd& point) const;

            void update_avg_return(const Eigen::ArrayXd& trial_return);

            void set_value(const Eigen::ArrayXd& new_value);

            Eigen::ArrayXd get_avg_return_or_value() const;

            double get_scalarised_avg_return_or_value(const Eigen::ArrayXd& weight) const;

            double radius() const;

            Eigen::ArrayXd center() const;

            double confidence_radius(const int total_backups_across_all_balls) const;

            double get_num_backups() const;
    };



    /**
     * CzBallList implementation
     * 
     * We make a slight optimisation by checking from smallest radius balls to largest radius balls when finding a relevant
     * ball
     * 
     * Member variables:
     *      
    */
    class CzBallList {
        protected:
            mutable std::mutex lock;
            int num_backups;
            int num_backups_before_allowed_to_split;
            double base_ball_radius;  // The largest radius (level 0) - initial ball
            int max_level;  // Maximum level (0 is largest, each level divides radius by 2)
            std::unordered_map<int,std::vector<std::shared_ptr<CzBall>>> ball_list;  // Maps level -> balls
            std::shared_ptr<CzBall> init_ball;
            int _dim;
            
            // Helper functions to convert between radius and level
            int radius_to_level(double radius) const;
            double level_to_radius(int level) const;

        public:
            /**
             * Constructor
             * 
             * Compute centroid of simplex of weights
             * Compute radius for initial ball (using simplex corner point)
             * Make a init ball with centroid point
             * 
             * Initialise member variables
            */
            CzBallList(int dim, int num_trials_before_allowed_to_split);
            
            CzBallList(const CzBallList&) = delete;
            CzBallList& operator=(const CzBallList&) = delete;
            CzBallList(CzBallList&&) = delete;
            CzBallList& operator=(CzBallList&&) = delete;

            /**
             * Gets the initial ball
            */
            std::shared_ptr<CzBall> get_init_ball() const;

            /**
             * Gets a list of all balls
            */
            std::shared_ptr<std::vector<std::shared_ptr<CzBall>>> get_all_balls() const;

            /**
             * Get most relevant balls
             * Recall the domain of larger balls excludes the domain of smaller balls, so can return when we find any 
             * relevant balls
             * That is, all the relevant balls will have the same radii
            */
            std::shared_ptr<std::vector<std::shared_ptr<CzBall>>> get_relevant_balls(Eigen::ArrayXd& weight) const;

            /**
             * Get a list of balls with radius above a certain length
            */
            std::shared_ptr<std::vector<std::shared_ptr<CzBall>>> get_balls_with_min_radius(double min_radius) const;

            /**
             * Returns a pretty print string of the ball list
            */
            std::string get_pretty_print_string() const;

            /**
             * Returns num backups
            */
            int get_num_backups() const;

            /**
             * Return a convex hull approximation
             */
            ConvexHull get_approximate_convex_hull() const;
            

        private:
            // Called by 'avg_return_update_ball_list' and 'set_value_update_ball_list'
            std::shared_ptr<CzBall> activate_new_ball_if_needed(
                const Eigen::ArrayXd& weight, 
                std::shared_ptr<CzBall> chosen_ball);

        public:
            /**
             * Update ball list
             * Using average returns
            */
            void avg_return_update_ball_list(
                const Eigen::ArrayXd& trial_return, 
                const Eigen::ArrayXd& weight, 
                std::shared_ptr<CzBall> chosen_ball);

            /**
             * Update ball list
             * Using average returns
            */
            void set_value_update_ball_list(
                const Eigen::ArrayXd& value, 
                const Eigen::ArrayXd& weight, 
                std::shared_ptr<CzBall> chosen_ball);
    };
}