#include "mo/data_structures/ball_list.h"

#include "mo/mo_helper.h"

#include <cmath>
#include <sstream>
#include <iostream>


using namespace std;

const double E = exp(1.0);
static double EPS = 1e-12;


namespace thts {
    CzBall::CzBall(const double radius, const Eigen::ArrayXd& center) :
        _radius(radius),
        _center(center),
        stats_lock(),
        num_backups(0),
        avg_return_or_value(Eigen::ArrayXd::Zero(center.size()))
    {
    }

    bool CzBall::point_in_domain(const Eigen::ArrayXd& point) const {
        return thts::helper::dist(_center, point) <= _radius + EPS;
    }

    void CzBall::update_avg_return(const Eigen::ArrayXd& trial_return) {
        lock_guard<mutex> lg(stats_lock);
        num_backups++;
        avg_return_or_value += (trial_return - avg_return_or_value) / (double) num_backups;
    }

    void CzBall::set_value(const Eigen::ArrayXd& new_value) {
        lock_guard<mutex> lg(stats_lock);
        num_backups++;
        avg_return_or_value = new_value;
    }

    Eigen::ArrayXd CzBall::get_avg_return_or_value() const {
        lock_guard<mutex> lg(stats_lock);
        return avg_return_or_value;
    }

    double CzBall::get_scalarised_avg_return_or_value(const Eigen::ArrayXd& weight) const {
        lock_guard<mutex> lg(stats_lock);
        return weight.matrix().dot(avg_return_or_value.matrix());
    }

    double CzBall::radius() const {
        return _radius;
    }

    Eigen::ArrayXd CzBall::center() const {
        return _center;
    }

    double CzBall::confidence_radius(const int total_backups_across_all_balls) const {
        lock_guard<mutex> lg(stats_lock);
        return log(total_backups_across_all_balls + E) / (1 + num_backups);
    }

    double CzBall::get_num_backups() const {
        lock_guard<mutex> lg(stats_lock);
        return num_backups;
    }

    // Helper functions to convert between radius and level
    // Level 0 = largest radius (base_ball_radius), each level divides by 2
    int CzBallList::radius_to_level(double radius) const {
        if (radius <= 0.0 || base_ball_radius <= 0.0) {
            throw runtime_error("Invalid radius or base_ball_radius for level conversion");
        }
        double ratio = base_ball_radius / radius;  // Note: base_ball_radius is larger
        // Round to nearest level to handle floating point precision
        int level = static_cast<int>(round(log2(ratio)));
        if (level < 0) {
            level = 0;  // Can't have negative levels
        }
        return level;
    }
    
    double CzBallList::level_to_radius(int level) const {
        if (level < 0) {
            throw runtime_error("Level cannot be negative");
        }
        return base_ball_radius / pow(2.0, level);
    }

    /**
     * Constructor
     * 
     * Compute centroid of simplex of weights
     * Compute radius for initial ball (using simplex corner point)
     * Make a init ball with centroid point
     * 
     * Initialise member variables
    */
    CzBallList::CzBallList(int dim, int num_backups_before_allowed_to_split) : 
        lock(),
        num_backups(0),
        num_backups_before_allowed_to_split(num_backups_before_allowed_to_split),
        base_ball_radius(0.0),
        max_level(0),
        ball_list(),
        init_ball(nullptr),
        _dim(dim)
    {
        if (dim <= 1) {
            throw runtime_error("Dimension must be positive");
        }

        Eigen::ArrayXd centroid(dim);
        for (int i=0; i<dim; i++) {
            centroid[i] = 1.0 / dim;
        }

        Eigen::ArrayXd simplex_corner_point = Eigen::ArrayXd::Zero(dim);
        simplex_corner_point[0] = 1.0;
        double init_ball_radius = thts::helper::dist(centroid,simplex_corner_point);

        init_ball = make_shared<CzBall>(init_ball_radius, centroid);
        
        lock_guard<mutex> lg(lock);
        base_ball_radius = init_ball_radius;  // This is the largest radius (level 0)
        max_level = 0;  // Initial ball is at level 0 (largest)
        ball_list[0].push_back(init_ball);
    }

    /**
     * Get init ball
    */
    shared_ptr<CzBall> CzBallList::get_init_ball() const {
        return init_ball;
    }

    /**
     * Get all balls
    */
    shared_ptr<vector<shared_ptr<CzBall>>> CzBallList::get_all_balls() const {
        shared_ptr<vector<shared_ptr<CzBall>>> all_balls = make_shared<vector<shared_ptr<CzBall>>>();
        lock_guard<mutex> lg(lock);
        for (int level = 0; level <= max_level; level++) {
            if (ball_list.find(level) != ball_list.end()) {
                for (shared_ptr<CzBall> ball : ball_list.at(level)) {
                    all_balls->push_back(ball);
                }
            }
        }
        return all_balls;
    }

    /**
     * Get most relevant balls
     * Recall the domain of larger balls excludes the domain of smaller balls, so can return when we find any 
     * relevant balls
     * That is, all the relevant balls will have the same radii
    */
    shared_ptr<vector<shared_ptr<CzBall>>> CzBallList::get_relevant_balls(
        Eigen::ArrayXd& weight) const 
    {
        shared_ptr<vector<shared_ptr<CzBall>>> relevant_balls;
        relevant_balls = make_shared<vector<shared_ptr<CzBall>>>();

        lock_guard<mutex> lg(lock);
        int current_max_level = max_level;
        // Iterate from smallest (max_level) to largest (level 0)
        for (int level = current_max_level; level >= 0; level--) {
            if (ball_list.find(level) != ball_list.end()) {
                for (shared_ptr<CzBall> ball : ball_list.at(level)) {
                    if (ball->point_in_domain(weight)) {
                        relevant_balls->push_back(ball);
                    }
                }
                if (relevant_balls->size() > 0) {
                    break;
                }
            }
        }

        if (relevant_balls->size() == 0) {
            throw runtime_error("Shouldn't get zero relevant balls unless something is wrong");
        }
        return relevant_balls;
    }

    /**
     * Get a list of balls with radius above a certain length
    */
    shared_ptr<vector<shared_ptr<CzBall>>> CzBallList::get_balls_with_min_radius(double min_radius) const {
        shared_ptr<vector<shared_ptr<CzBall>>> bigger_balls;
        bigger_balls = make_shared<vector<shared_ptr<CzBall>>>();

        lock_guard<mutex> lg(lock);
        int min_level = radius_to_level(min_radius);
        int current_max_level = max_level;
        // Iterate from min_level to level 0 (largest)   (smaller to larger radius)
        // But we want balls with radius >= min_radius, which means level <= min_level
        for (int level = min_level; level >= 0; level--) {
            if (ball_list.find(level) != ball_list.end()) {
                vector<shared_ptr<CzBall>> cur_level_balls = ball_list.at(level);
                bigger_balls->insert(bigger_balls->end(), cur_level_balls.begin(), cur_level_balls.end());
            }
        }

        return bigger_balls;
    }

    string CzBallList::get_pretty_print_string() const {
        stringstream ss;
        ss << "radius // ball_visits // avg_return // center" << endl;
        lock_guard<mutex> lg(lock);
        int current_max_level = max_level;
        // Iterate from largest (level 0) to smallest (max_level)  
        for (int level = 0; level <= current_max_level; level++) {
            if (ball_list.find(level) != ball_list.end()) {
                for (shared_ptr<CzBall> ball : ball_list.at(level)) {
                    ss << ball->radius() << " // " << ball->get_num_backups() << " // [";
                    Eigen::ArrayXd ar = ball->get_avg_return_or_value();
                    for (int i=0; i<ar.size(); i++) {
                        ss << ar[i] << ",";
                    }
                    ss << "] // [";
                    Eigen::ArrayXd c = ball->center();
                    for (int i=0; i<c.size(); i++) {
                        ss << c[i] << ",";
                    }
                    ss << "]" << endl;
                }
            }
        }
        return ss.str();
    }

    int CzBallList::get_num_backups() const {
        lock_guard<mutex> lg(lock);
        return num_backups;
    }
    
    shared_ptr<CzBall> CzBallList::activate_new_ball_if_needed(
        const Eigen::ArrayXd& weight, 
        shared_ptr<CzBall> chosen_ball) 
    {
        if (chosen_ball->get_num_backups() >= num_backups_before_allowed_to_split 
            && chosen_ball->confidence_radius(num_backups) <= chosen_ball->radius())
        {   
            double new_ball_radius = chosen_ball->radius() / 2.0;
            chosen_ball = make_shared<CzBall>(new_ball_radius, weight);
            
            lock_guard<mutex> lg(lock);
            int new_level = radius_to_level(new_ball_radius);
            ball_list[new_level].push_back(chosen_ball);
            if (new_level > max_level) {
                max_level = new_level;  // Higher level means smaller radius
            }
        }
        return chosen_ball;
    }

    /**
     * Update ball list
     * Using average returns
    */
    void CzBallList::avg_return_update_ball_list(
        const Eigen::ArrayXd& trial_return, 
        const Eigen::ArrayXd& weight, 
        shared_ptr<CzBall> chosen_ball) 
    {
        {
            lock_guard<mutex> lg(lock);
            num_backups++;
        }
        chosen_ball = activate_new_ball_if_needed(weight, chosen_ball);
        chosen_ball->update_avg_return(trial_return);
    }

    /**
     * Update ball list
     * Using average returns
    */
    void CzBallList::set_value_update_ball_list(
        const Eigen::ArrayXd& value, 
        const Eigen::ArrayXd& weight, 
        shared_ptr<CzBall> chosen_ball) 
    {
        {
            lock_guard<mutex> lg(lock);
            num_backups++;
        }
        chosen_ball = activate_new_ball_if_needed(weight, chosen_ball);
        chosen_ball->set_value(value);
    }

    /**
     * Gets an approximate convex hull from this ball list
     * 
     * Originally was just taking MO over all values from all balls, but there 
     * is the case that some coarse balls have stale values that "got lucky"
     * 
     * Think the most fair way to do this is to generate a bunch of random 
     * weights, and uses the balls that would be used in CZ during 
     * recommendations
     *
     * Note that there is one ball list per action, and CZT node will take the 
     * union of all the convex hulls from all the ball lists
     */
    ConvexHull CzBallList::get_approximate_convex_hull() const 
    {
        int num_random_weights = 128;
        vector<Eigen::ArrayXd> random_weights = thts::helper::get_well_spaced_simplex_points(
            num_random_weights, _dim);

        unordered_set<Vec> ball_avg_returns;
        for (Eigen::ArrayXd& random_weight : random_weights) {
            shared_ptr<vector<shared_ptr<CzBall>>> relevant_balls = get_relevant_balls(random_weight);
            
            Eigen::ArrayXd most_relevant_return = Eigen::ArrayXd::Zero(_dim);
            double most_relevant_return_value = numeric_limits<double>::lowest();
            for (shared_ptr<CzBall> ball : *relevant_balls) {
                double ball_value = ball->get_scalarised_avg_return_or_value(random_weight);
                if (ball_value > most_relevant_return_value) {
                    most_relevant_return_value = ball_value;
                    most_relevant_return = ball->get_avg_return_or_value();
                }
            }
            
            ball_avg_returns.insert(most_relevant_return);
        }
        return ConvexHull(ball_avg_returns);
    }

}