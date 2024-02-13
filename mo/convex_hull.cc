#include "mo/convex_hull.h"

#include "helper_templates.h"

#include <iostream>
#include <stdexcept>

#include "ortools/linear_solver/linear_solver.h"



/**
 * ConvexHull implementation
 */
namespace thts {
    using namespace std;
    
    /**
     * Constructor, empty
    */
    template <typename T>
    ConvexHull<T>::ConvexHull() :
        ch_points()
    {
    };

    /**
     * Constructor, adding points immediately
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const vector<pair<Eigen::ArrayXd,T>>& init_points) :
        ch_points()
    {  
        unordered_set<TaggedPoint<T>> tagged_init_points;
        tagged_init_points.reserve(init_points.size());
        for (const pair<Eigen::ArrayXd,T>& pr : init_points) {
            tagged_init_points.emplace(pr.first,pr.second);
        }
        ch_points = prune(tagged_init_points);
    };

    /**
     * Constructor, adding points immediately, with one tag
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const vector<Eigen::ArrayXd>& init_points, const T& tag) :
        ch_points()
    {  
        unordered_set<TaggedPoint<T>> tagged_init_points;
        tagged_init_points.reserve(init_points.size());
        for (const Eigen::ArrayXd& point : init_points) {
            tagged_init_points.insert(TaggedPoint<T>(point,tag));
        }
        ch_points = prune(tagged_init_points);
    };
    
    /**
     * Constructor, set of Tagged points
     * Note that not assuming that 'init_points' is a Pareto Front, do use 'add_points' function to prune
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const unordered_set<TaggedPoint<T>>& init_points, bool already_pareto_front) :
        ch_points(already_pareto_front ? init_points : prune(init_points))
    {
    };

    /**
     * Copy constructor
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const ConvexHull<T>& ch) :
        ch_points(ch.ch_points) 
    {
    };

    /**
     * Move constructor
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const ConvexHull<T>&& ch) :
        ch_points(move(ch.ch_points)) 
    {
    };
    

    /**
     * Retuns if 'point' is dominated by any point in 'ref_points'. We're using google's OR tools here
     * https://developers.google.com/optimization
     * 
     * Mostly because it provides a really clean interface
     * 
     * TODO: consider if bothered by speed here using a faster linear solver, when googling, google's OR tools isnt the 
     *      first option to come up, and something like 'https://www.alglib.net/download.php' seems more typical
     * 
     * 
     * Method is adapted from:
     * http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/roijersjair15.pdf
     * 
     * Prunes points that are not on the convex hull fronts. A point p should be pruned from a set of points ps if the 
     * following linear program returns a negative value:
     * 
     * max x = ( 0 ... 0 1)^T  ( w x ) 
     * s.t. w^T (p - p') - x >= 0 for all p' \in ps
     *      \sum_i w_i = 1
     * 
     * This is implemented using the following constraints: 
     * 
     * (  p-p_1       -1 )       (   )         (   )
     * (   ...        ...)   *   ( w )   >=    ( 0 )
     * (   ...        ...)       (   )         (   )
     * (  p-p_n       -1 )       ( x )         ( 0 )
     * 
     * (  1 ... 1      0 )   *   ( w x )^T   =   1
     * 
     * Objective is given by:
     *       max (0 ... 0 1) * ( w x )^T
     * 
     * where p_1, ..., p_n are the points NOT including p in pts. 
     * 
     * Bounds are 0 <= w_i <= 1 and x is arbitrary
     * 
     * Note that if p is in ps, then there is a constraint that reduces to '-x >= 0', which will force x to be negative
     * This would be fine if this was 'weakly_convex_dominated' but because we want strong domination, if 
     * 'ref_points' contains a point equal to 'point', when we ignore it.
     */
    bool strongly_convex_dominated(const unordered_set<TaggedPoint<T>>& ref_points, TaggedPoint<T>& point)
    {  
        // Make solver, and get variable for inf in setting bounds
        unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
        const double inf = solver->infinity();
        if (!solver) {
            throw runtime_error("GLOP linear solver not available for convex hull pruning");
        }

        // Get n (number of points in 'ref_points' and dimension of vectors)
        int n = ref_points.size();
        int dim = point.point.size();

        // Add variables for w and x
        vector<MPVariable*>> w_vars;
        for (int i=0; i<dim; i++) {
            stringstream ss;
            ss << "w_" << i;
            MPVariable* w_i_var = solver->MakeNumVar(0.0, 1.0, ss.str());
            w_vars.push_back(w_i_var);
        }
        MPVariable* x_var = solver->MakeNumVar(-inf, inf, "x");

        // Add row constrains for the inequality constraint above
        for (TaggedPoint<T>& ref_p : ref_points) {
            if (ref_p == point) continue;
            Eigen::ArrayXd diff = point.point - ref_p.point; // p-p_k

            MPConstraint* c = solver->MakeRowConstraint(0.0, inf); // >= 0
            for (int i=0; i<dim; i++) {
                c->SetCoefficient(w_vars[i], diff[i]); // + (p-p_k)_i * w_i
            }

            c->SetCoefficient(x_var, -1.0); // + -1 * x
        }

        // Add row constraint for the equality constraint
        MPConstraint* c = solver->MakeRowConstraint(1.0, 1.0); // == 1.0
        for (int i=0; i<dim; i++) {
            c->SetCoefficient(w_vars[i], 1.0); // + 1.0 * w_i
        }

        // Set objective (max x)
        MPObjective* obj = solver->MutableObjective();
        objective->SetCoefficient(x_var, 1.0);
        objective->SetMaximization();

        // Solve 
        MPSolver::ResultStatus result_status = solver->Solve();
        if (result_state != MPSolver::OPTIMAL) {
            throw runtime_error("Linear solver is saying linear program doesnt have an optimal solution");
        }

        // Check if optimal value was negative (meaning its dominated) or not
        return obj->Value() < 0.0;
    };

    /**
     * Returns the set of points from 'points' that are not dominated by any points in 'ref_points'
    */
    template <typename T>
    unordered_set<TaggedPoint<T>> ConvexHull<T>::prune(
        const unordered_set<TaggedPoint<T>>& ref_points, const unordered_set<TaggedPoint<T>>& points) const
    {
        if (ref_points.size() == 0 || points.size() == 0) {
            return unordered_set<TaggedPoint<T>>(points);
        }

        unordered_set<TaggedPoint<T>> new_set;
        new_set.reserve(points.size());

        for (const TaggedPoint<T>& p : points) {
            if (!strongly_convex_dominated(ref_points, p)) {
                new_set.insert(p);
            }
        }
        
        return new_set;
    };

    /**
     * Prunes a set of 'points' to a set of points that form a Convex Hull
    */
    template <typename T>
    unordered_set<TaggedPoint<T>> ConvexHull<T>::prune(const unordered_set<TaggedPoint<T>>& points) const {
        unordered_set<TaggedPoint<T>> pruned_points(points);
        
        for (auto it = pruned_points.begin(); it != pruned_points.end(); ) {
            bool is_dominated = strongly_convex_dominated(pruned_points, *it);
            if (is_dominated) {
                it = pruned_points.erase(it);
            } else {
                it++;
            }
        }

        return pruned_points;
    }

    /**
     * Copied from mo/pareto_front.cc
    */
    template<typename T>
    size_t ConvexHull<T>::size() const {
        return ch_points.size();
    }

    /**
     * Copied from mo/pareto_front.cc
    */
    template<typename T>
    void ConvexHull<T>::set_tags(const T& new_tag) {
        unordered_set<TaggedPoint<T>> new_ch_points;
        new_ch_points.reserve(size());
        for (const TaggedPoint<T>& point : ch_points) {
            new_ch_points.insert(TaggedPoint<T>(point.point, new_tag));
        }
        ch_points = move(new_ch_points);
    }

    /**
     * Copied from mo/pareto_front.cc
    */
    template <typename T>
    ConvexHull<T> ConvexHull<T>::scale(double scale) const
    {
        unordered_set<TaggedPoint<T>> scaled_ch_points;
        scaled_ch_points.reserve(size());
        for (const TaggedPoint<T>& point : ch_points) {
            scaled_ch_points.insert(TaggedPoint<T>(point.point*scale, point.tag));
        }
        return ConvexHull<T>(scaled_ch_points, true);
    };

    /**
     * Copied from mo/pareto_front.cc
    */
    template <typename T>
    ConvexHull<T> ConvexHull<T>::combine(const ConvexHull<T>& other) const {
        if (other.size() > size()) {
            return other.combine(*this);
        }
        if (size() == 0) {
            return ConvexHull<T>(other);
        } else if (other.size() == 0) {
            return ConvexHull<T>(*this);
        }
        
        // as already pareto fronts, only need to check if points dominated by the other pareto front
        // care here if this and other contain the same vector, as using weak Pareto domination
        unordered_set<TaggedPoint<T>> pruned_points_one = prune(other.ch_points, ch_points);
        unordered_set<TaggedPoint<T>> pruned_points_two = prune(pruned_points_one, other.ch_points);
        
        pruned_points_one.reserve(pruned_points_one.size() + pruned_points_two.size());
        for (const TaggedPoint<T>& point : pruned_points_two) {
            pruned_points_one.insert(point);
        }

        return ConvexHull<T>(pruned_points_one, true);
    };

    /**
     * Copied from mo/pareto_front.cc
    */
    template <typename T>
    ConvexHull<T> ConvexHull<T>::add(const ConvexHull<T>& other) const 
    {
        if (ch_points.size() == 0) {
            return ConvexHull<T>(other);
        } else if (other.ch_points.size() == 0) {
            return ConvexHull<T>(*this);
        }

        unordered_set<TaggedPoint<T>> summed_points;
        for (const TaggedPoint<T>& point : ch_points) {
            for (const TaggedPoint<T>& other_point : other.ch_points) {
                summed_points.insert(TaggedPoint<T>(point.point + other_point.point, point.tag));
            }
        }

        // constructor will prune points
        return ConvexHull<T>(summed_points);
    };

    /**
     * Copied from mo/pareto_front.cc
    */
    template <typename T>
    ConvexHull<T> ConvexHull<T>::add(const Eigen::ArrayXd& v) const 
    {
        unordered_set<TaggedPoint<T>> summed_points;
        for (const TaggedPoint<T>& point : ch_points) {
            summed_points.insert(TaggedPoint<T>(point.point + v, point.tag));
        }
        return ConvexHull<T>(summed_points, true);
    };
}

/**
 * These are all copied from ParetoFront - see them for doc
*/
namespace std {
    using namespace thts;

    template <typename T>
    ConvexHull<T> operator*(const ConvexHull<T>& ch, double s) {
        return ch.scale(s);
    }
    
    template <typename T>
    ConvexHull<T> operator*(double s, const ConvexHull<T>& ch) {
        return ch.scale(s);
    }

    template <typename T>
    ConvexHull<T> operator|(const ConvexHull<T>& ch1, const ConvexHull<T>& ch2) {
        return ch1.combine(ch2);
    }

    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& ch1, const ConvexHull<T>& ch2) {
        return ch1.add(ch2);
    }

    template <typename T>
    ConvexHull<T> operator+(const ConvexHull<T>& ch, const Eigen::ArrayXd& v) {
        return ch.add(v);
    }

    template <typename T>
    ConvexHull<T> operator+(const Eigen::ArrayXd& v, const ConvexHull<T>& ch) {
        return ch.add(v);
    }

    template <typename T>
    ostream& operator<<(ostream& os, const ConvexHull<T>& ch) {
        os << "ConvexHull = {" << endl;
        for (TaggedPoint<T> point : ch.ch_points) {
            os << point << endl;
        }
        os << "}";
        return os;
    }
}