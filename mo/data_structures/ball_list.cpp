#include "mo/data_structures/ball_list.h"

#include "mo/mo_helper.h"

#include <cmath>
#include <sstream>


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
        largest_ball_radius(0.0),
        smallest_ball_radius(0.0),
        ball_list(),
        init_ball(nullptr)
    {
        Eigen::ArrayXd centroid(dim);
        for (int i=0; i<dim; i++) {
            centroid[i] = 1.0 / dim;
        }

        Eigen::ArrayXd simplex_corner_point = Eigen::ArrayXd::Zero(dim);
        simplex_corner_point[0] = 1.0;
        double init_ball_radius = thts::helper::dist(centroid,simplex_corner_point);

        init_ball = make_shared<CzBall>(init_ball_radius, centroid);
        
        lock_guard<mutex> lg(lock);
        largest_ball_radius = init_ball_radius;
        smallest_ball_radius = init_ball_radius;
        ball_list[init_ball_radius].push_back(init_ball);
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
        shared_ptr<vector<shared_ptr<CzBall>>> all_balls;        
        for (double cur_radius = smallest_ball_radius; cur_radius <= largest_ball_radius; cur_radius *= 2.0) {
            lock_guard<mutex> lg(lock);
            for (shared_ptr<CzBall> ball : ball_list.at(cur_radius)) {
                all_balls->push_back(ball);
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

        for (double cur_radius = smallest_ball_radius; cur_radius <= largest_ball_radius; cur_radius *= 2.0) {
            lock_guard<mutex> lg(lock);
            for (shared_ptr<CzBall> ball : ball_list.at(cur_radius)) {
                if (ball->point_in_domain(weight)) {
                    relevant_balls->push_back(ball);
                }
            }
            if (relevant_balls->size() > 0) {
                break;
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

        for (double cur_radius = min_radius; cur_radius <= largest_ball_radius; cur_radius *= 2.0) {
            lock_guard<mutex> lg(lock);
            vector<shared_ptr<CzBall>> cur_radius_balls = ball_list.at(cur_radius);
            bigger_balls->insert(bigger_balls->end(), cur_radius_balls.begin(), cur_radius_balls.end());
        }

        return bigger_balls;
    }

    string CzBallList::get_pretty_print_string() const {
        stringstream ss;
        ss << "radius // ball_visits // avg_return // center" << endl;
        for (double cur_radius = largest_ball_radius; cur_radius >= smallest_ball_radius; cur_radius /= 2.0) {
            lock_guard<mutex> lg(lock);
            for (shared_ptr<CzBall> ball : ball_list.at(cur_radius)) {
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
            ball_list[new_ball_radius].push_back(chosen_ball);
            if (new_ball_radius < smallest_ball_radius) {
                smallest_ball_radius = new_ball_radius;
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
     */
    ConvexHull CzBallList::get_approximate_convex_hull() const 
    {
        throw runtime_error("Approximate convex hull from ball list not written yet");
        return ConvexHull();
    }

}