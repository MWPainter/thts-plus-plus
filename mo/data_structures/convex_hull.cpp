#include "mo/data_structures/convex_hull.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"
#include "mo/mo_thts_types.h"

#include <limits>

#include <iostream>
#include <stdexcept>

// #include "lemon/lp.h"
#include "coin-or/ClpSimplex.hpp"
#include "coin-or/CoinPackedMatrix.hpp"


#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/RboxPoints.h"


/**
 * ConvexHull implementation
 */
namespace thts {
    using namespace std;
    // using namespace lemon;
    
    /**
     * Helper function to remove approximate duplicates from a set of Vec
     * Keeps the first occurrence of each approximately equal vector
     */
    static unordered_set<Vec> remove_approx_duplicates(const unordered_set<Vec>& points) {
        unordered_set<Vec> deduplicated;
        deduplicated.reserve(points.size());
        
        for (const Vec& point : points) {
            bool is_duplicate = false;
            for (const Vec& existing : deduplicated) {
                if (point.approx_equals(existing)) {
                    is_duplicate = true;
                    break;
                }
            }
            if (!is_duplicate) {
                deduplicated.insert(point);
            }
        }
        
        return deduplicated;
    }
    
    /**
     * Constructor, empty
    */
    ConvexHull::ConvexHull() :
        ch_points()
    {
    };
    
    /**
     * Constructor, set of Tagged points
     * Note that not assuming that 'init_points' is a Pareto Front, do use 'add_points' function to prune
    */
    ConvexHull::ConvexHull(const unordered_set<Vec>& init_points, bool already_convex_hull) :
        ch_points(already_convex_hull ? init_points : prune(remove_approx_duplicates(init_points)))
    {
    };

    /**
     * Constructor initialising from a heuristic val
    */
    ConvexHull::ConvexHull(const Vec& heuristic_val) :
        ch_points()
    {
        ch_points.insert(heuristic_val);
    };

    /**
     * Copy constructor
    */
    ConvexHull::ConvexHull(const ConvexHull& ch) :
        ch_points(ch.ch_points) 
    {
    };

    /**
     * Move constructor
    */
    ConvexHull::ConvexHull(const ConvexHull&& ch) :
        ch_points(std::move(ch.ch_points)) 
    {
    };

    /**
     * Copy assignment
    */
    ConvexHull& ConvexHull::operator=(const ConvexHull& ch) 
    {
        this->ch_points = ch.ch_points;
        return *this;
    }

    /**
     * Move assignment
    */
    ConvexHull& ConvexHull::operator=(const ConvexHull&& ch) 
    {
        this->ch_points = std::move(ch.ch_points);
        return *this;
    }

    /**
     * *=
     * 
     * Saw it mentiond in this SO post: https://stackoverflow.com/questions/14918790/overloading-the-operator
     * A better implementation of operator+ and operator+= would define operator+= first, as that can modify 
     * a reference, rather than making copies, and then implemnt operator+ as "return lhs += rhs".
     * 
     * Think as we've defined move assignment stuff, that our implementation should be just fine
     * 
     * But a TODO could be to check this out? Any maybe use std::shared_ptr<std::unordered_set<Vec>> for 
     * ch_points
     * 
    */
    ConvexHull& ConvexHull::operator*=(double rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs * rhs;
        return lhs;
    }

    ConvexHull& ConvexHull::operator*=(const Vec& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs * rhs;
        return lhs;
    }

    /**
     * |=
    */
    ConvexHull& ConvexHull::operator|=(const ConvexHull& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs | rhs;
        return lhs;
    }
    ConvexHull& ConvexHull::operator|=(const ConvexHull&& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs | rhs;
        return lhs;
    }

    /**
     * +=
    */
    ConvexHull& ConvexHull::operator+=(const ConvexHull& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs + rhs;
        return lhs;
    }
    ConvexHull& ConvexHull::operator+=(const ConvexHull&& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs + rhs;
        return lhs;
    }
    ConvexHull& ConvexHull::operator+=(const Vec& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs + rhs;
        return lhs;
    }
    ConvexHull& ConvexHull::operator-=(const Vec& rhs)
    {
        ConvexHull& lhs = *this;
        lhs = lhs - rhs;
        return lhs;
    }

    

    /**
     * Retuns if 'point' is dominated by any point in 'ref_points'. 
     * 
     * Method is adapted from:
     * http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/roijersjair15.pdf
     * 
     * Prunes points that are not on the convex hull fronts. A point p should be pruned from a set of points ps if the 
     * following linear program returns a negative value:
     * 
     * max x = ( 0 ... 0 1)  ( w x )^T 
     * s.t. w^T (p - p') - x >= 0 for all p' \in ps
     *      \sum_i w_i = 1
     * 
     * This is implemented using the following constraints: 
     * 
     * (  p-p_1       -1 )       (   )   >=    ( 0 )
     * (   ...        ...)   *   (   )   >=    (...)        
     * (   ...        ...)       ( w )   >=    (...)
     * (  p-p_n       -1 )       (   )   >=    ( 0 )
     * (  1 ... 1      0 )   *   ( x )   ==    ( 1 )         
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
     * 
     * For the boundary case, when x == 0, we're going to say it's dominated. Consider the points (0,2), (1,1) and (2,0)
     * the point (1,1) we want to be dominated, and it's optimal value of x will be zero (found at w=0.5*(1,1))
     * 
     * If ref_points.size() == 0, or ref_points == {point}, then there will be no constraint to bound the value of x 
     * and an error will be thrown. So catch these cases at the start. In this case there is no point in 'ref_points' 
     * to dominate 'point', so return false;
     * 
     * TODO: probably should just call this "convex_dominated", dont think this needs a notion of strong/weak. 
     * Generally weak/strong only considers domination of points with itself, i.e. if p == p1 == p2, then p1 weakly 
     * dominates p2, but doesnt strongly dominate p2
     * 
     * Going to be using CLP to solve these programs from https://github.com/coin-or/Clp/tree/master. (Originally 
     * tried to use lemon with GLPK, but that was buggy, and couldn't get lemon working with any other backend).
     * 
     * From the readme, CLP solves linear programs of the form:
     * minimize c^Tx 
     * such that lhs ≤ Ax ≤ rhs 
     * and lb ≤ x ≤ ub
     * 
     * There's a bit of weirdness, because we have to build A as a 'sparse matrix', although in our case it will be 
     * dense. They also use "number of rows" and "number of collumns" in (from my perspective) a bit of a confusing 
     * way, from what I can gather, number of collumns = number of vars, number of rows = number of constraints.
     * 
     * Also N.B. going to implement sum_i w_i = 1 constraint (last constraint/row above) as the first constraint/row
     * 
     * Some notes on making the dense CoinPackedMatrix, as we have to tell it that it's dense matrix:
     * - let A be an nxm matrix, so n constraints and m variables
     * - coin uses a 1D array to specify the elements of the matrix, because they can be sparse
     * - matrix_row_start gives the indices that delimit the rows of the matrix in the 1D array 
     * -- so this will be matrix_row_start = {0,m,2*m,...,n*m} for us
     * - column specifies which column (varaible) each element in the array corresponds to
     * -- for use this will look like {{0,1,...,m-1},{0,1,...,m-1},...,{0,1,...,m-1}}
     * 
     * Finally, note that CLP will minimise the objective, and the program we gave above is maximisation
     *
     * Also, assumes that any approximately equal points have been removed from any convex hull sets already. 
     * If point is in ref_points, then we are pruning a convex hull, and that ref point should be ignored.
     */
    bool ConvexHull::strongly_convex_dominated( 
        const unordered_set<Vec>& ref_points, 
        const Vec& point) // static
    {   
        // Base/degenerate case
        // Point cant be dominated by nothing
        if (ref_points.size() == 0) {
            return false;
        }

        // Base/degenerate case
        // If ref_points contains only one point, revert to pareto domination
        // Unless we are in the case that have one point in a convex hull 
        if (ref_points.size() == 1) {
            if (ref_points.contains(point)) {
                return false;
            }
            Vec ref_point = *ref_points.begin();
            return ref_point.weakly_pareto_dominates(point);
        }

        // Size of lp
        int dim = point.vec.size();
        int num_vars = dim+1;
        int num_constraints = ref_points.size() + 1;
        if (ref_points.contains(point)) {
            num_constraints--;
        }

        // Variable bounds
        double var_lower_bound[num_vars];
        double var_upper_bound[num_vars];
        
        for (int i=0; i<num_vars-1; i++) {
            var_lower_bound[i] = 0.0;
            var_upper_bound[i] = 1.0;
        }
        var_lower_bound[num_vars-1] = -COIN_DBL_MAX;
        var_upper_bound[num_vars-1] = COIN_DBL_MAX;

        // Constraints lower and upper bounds
        double constraint_lower_bound[num_constraints];
        double constraint_upper_bound[num_constraints];
        
        for (int i=0; i<num_constraints-1; i++) {
            constraint_lower_bound[i] = 0.0;
            constraint_upper_bound[i] = COIN_DBL_MAX;
        }
        constraint_lower_bound[num_constraints-1] = 1.0;
        constraint_upper_bound[num_constraints-1] = 1.0;

        // Construct a 'CoinPackedMatrix' for the linear program
        // Build arrays to tell coin this is a dense matrix
        // See comments above
        int matrix_num_elements = num_vars * num_constraints;

        CoinBigIndex matrix_row_start[num_constraints+1];
        int column[matrix_num_elements];

        for (int i=0; i<num_constraints; i++) {
            matrix_row_start[i] = i*num_vars;

            for (int j=0; j<num_vars; j++) {
                int index = i*num_vars + j;
                column[index] = j;
            }
        }
        matrix_row_start[num_constraints] = matrix_num_elements;

        // Finally, make the 1D array that specifies the matrix
        double matrix[matrix_num_elements];

        int index = 0;
        for (const Vec& ref_p : ref_points) {
            if (ref_p == point) continue;
            Eigen::ArrayXd diff = point.vec - ref_p.vec; // p-p_k
            for (int i=0; i<diff.size(); i++) {
                matrix[index++] = diff[i];
            }
            matrix[index++] = -1.0;
        }

        for (int i=0; i<dim; i++) {
            matrix[index++] = 1.0;
        }
        matrix[index] = 0.0;
        
        CoinPackedMatrix coin_packed_matrix(
            false, num_vars, num_constraints, matrix_num_elements, matrix, column, matrix_row_start, nullptr);

        // Define the objective (to minimise) and the linear program (simplex)
        // N.B. the true param to 'lp' should suppress CLP printing to stdout
        // And tell clp to not print anything out
        ClpSimplex lp;
        lp.setLogLevel(0);
        double objective_coeffs[num_vars];

        for (int i=0; i<dim; i++) {
            objective_coeffs[i] = 0.0;
        }
        objective_coeffs[dim] = -1.0;

        lp.loadProblem(
            coin_packed_matrix, 
            var_lower_bound, 
            var_upper_bound, 
            objective_coeffs, 
            constraint_lower_bound, 
            constraint_upper_bound);

        // Solve
        // Can call primal or dual and read out primal solution (which is what we want)
        // Not sure if there's any difference in implementations there *shrugs* but this is what the examples call so...
        lp.dual();

        // // For debugging
        // cout << "In lp solver:" << endl;
        // cout << "Obj values = " << lp.objectiveValue() << endl;
        // const double *soltn = lp.primalColumnSolution();
        // for (int i=0; i<num_vars; i++) {
        //     cout << "Var[" << i << "] = " << soltn[i] << endl;
        // }
        // cout << endl;

        // Also seems like should use lp.numberPrimalInfeasibilities() to check actually got a solution
        // if (lp.numberPrimalInfeasibilities() > 0) {
        // If infeasible, return error, shouldn't be passing infeasible problems to lp solver
        if (!lp.primalFeasible()) {
            
            // For debugging
            cout << "In lp solver:" << endl;
            cout << "Obj values = " << lp.objectiveValue() << endl;
            const double *soltn = lp.primalColumnSolution();
            for (int i=0; i<num_vars; i++) {
                cout << "Var[" << i << "] = " << soltn[i] << endl;
            }
            cout << "Number of primal infeasibilities = " << lp.numberPrimalInfeasibilities() << endl;
            cout << endl;

            // Print out variables run with
            cout << "Ref points = " << thts::helper::unordered_set_pretty_print_string(ref_points) << endl;
            cout << "Point to consider = " << point << endl;

            throw runtime_error("Lin prog in convex hull cant be solved, its infeasible or unbounded.");
        }

        // Read out result, objective in CLP is minimise -x
        double x = -lp.objectiveValue();
        return (x <= 0.0);
    };

    /**
     * Below is the original lemon implementation. Dont want to delete it because the interface is so nice and I wish 
     * it worked :( 
     * 
     * TODO: add documentation for the below
     * 
     * For the LP solver, we use lemon https://lemon.cs.elte.hu/pub/tutorial/a00020.html, because it provides a really 
     * clean symbolic interface to use. Really its a wrapper around other packages such as GLPK, Clp, Cbc, ILOG CPLEX 
     * and SoPlex https://lemon.cs.elte.hu/trac/lemon/wiki/InstallLinux
     */
    // template <typename T>
    // bool ConvexHull::strongly_convex_dominated(
    //     const unordered_set<Vec>& ref_points, 
    //     const Vec& point) const
    // {   
    //     // Base case where lp will be unbounded and would throw an error
    //     if (ref_points.size() == 0 || (ref_points.size() == 1 && ref_points.contains(point))) {
    //         return false;
    //     }

    //     // Make lp
    //     // Get n (number of points in 'ref_points' and dimension of vectors)
    //     lemon::Lp lp;
    //     int dim = point.point.size();

    //     // Add variables for w and x
    //     vector<lemon::Lp::Col> w;
    //     for (int i=0; i<dim; i++) {
    //         w.push_back(lp.addCol());
    //         lp.colLowerBound(w[i], 0.0);
    //         lp.colUpperBound(w[i], 1.0);
    //     }
    //     lemon::Lp::Col x = lp.addCol();

    //     // Add row constrains for the inequality constraint above (take care to not include 'point')
    //     for (const Vec& ref_p : ref_points) {
    //         if (ref_p == point) continue;
    //         Eigen::ArrayXd diff = point.point - ref_p.point; // p-p_k

    //         lemon::Lp::Expr row_expr = 0;
    //         for (int i=0; i<dim; i++) {
    //             row_expr += diff[i] * w[i];
    //         }
    //         row_expr += -1.0 * x;
    //         lemon::Lp::Constr row_constr = (row_expr >= 0.0);
    //         lp.addRow(row_constr);
    //     }

    //     // Add row constraint for the equality constraint
    //     lemon::Lp::Expr row_expr = 0;
    //     for (int i=0; i<dim; i++) {
    //         row_expr += w[i];
    //     };
    //     lemon::Lp::Constr row_constr = (row_expr == 1.0);
    //     lp.addRow(row_constr);

    //     // Set objective (max x)
    //     lp.max();
    //     lp.obj(x);

    //     // Solve 
    //     lp.solve();
    //     if (lp.primalType() != lemon::Lp::OPTIMAL) {
    //         // cout << "Lemon is saying it didn't find optimal solution. From testing these cases it seems like this "
    //         //     << "usually happens when it is actually feasible and should be returning true. For now, going to "
    //         //     << "hackily just return true here. Here are the points for reference:"; 
    //         // cout << "Point considering being pruned = " << point << endl;
    //         // cout << "And set of reference points = " 
    //         //      << thts::helper::unordered_set_pretty_print_string(ref_points) << endl;
    //         // return true;

    //         cout << "Getting error in linear programming solver." << endl;
    //         cout << "Point considering being pruned = " << point << endl;
    //         cout << "And set of reference points = " 
    //              << thts::helper::unordered_set_pretty_print_string(ref_points) << endl;
    //         cout << "And lp.primalType() == " << lp.primalType() << endl;
    //         throw runtime_error("Lin prog in convex hull cant be solved. If not optimal its infeasible or unbounded");
    //     }

    //     // Check if optimal value was negative (meaning its dominated) or not
    //     return lp.primal() <= 0.0;
    // };

    /**
     * Prunes a set of 'points' to a set of points that form a Pareto Front
    */
    unordered_set<Vec> ConvexHull::pareto_prune(const unordered_set<Vec>& points) // static
    {
        unordered_set<Vec> pruned_points;
        pruned_points.reserve(points.size());
        vector<Vec> points_vec(points.begin(), points.end());
        for (size_t i=0; i<points_vec.size(); i++) {
            bool is_dominated = false;
            for (size_t j=0; j<points_vec.size(); j++) {
                if (i == j) {
                    continue;
                }
                if (points_vec[j].strongly_pareto_dominates(points_vec[i])) {
                    is_dominated = true;
                    break;
                }
            }
            if (!is_dominated) {
                pruned_points.insert(points_vec[i]);
            }
        }
        return pruned_points;
    }

    /**
     * Prunes a set of 'points' to a set of points that form a Convex Hull
     * 
     * Because working with a single set of points, 'pruned_points' will always contain *it in the 
     * strongly_convex_dominated call, so set 'ignore_if_point_in_ref_points' to true here, to avoid all points being 
     * pruned by themselves
     * 
     * The linear program pruning seems to struggle when two points are colinear along one of the axes
     * So first pareto prune the points to solve these cases
    */
    unordered_set<Vec> ConvexHull::prune(const unordered_set<Vec>& points) // static
    {
        unordered_set<Vec> pruned_points = pareto_prune(points);
        
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
     * Get the dimension of vectors in the convex hull, assuming it's not empty
     * Returns -1 if it is empty
     */
    int ConvexHull::reward_dim() const {
        if (ch_points.size() == 0) {
            return -1;
        }
        return ch_points.begin()->dim();
    }

    /**
     * Copied from mo/pareto_front.cc
    */
    size_t ConvexHull::size() const {
        return ch_points.size();
    }

    /**
     * Copied from mo/pareto_front.cc
    */
    ConvexHull ConvexHull::scale(double scale) const
    {
        unordered_set<Vec> scaled_ch_points;
        scaled_ch_points.reserve(size());
        for (const Vec& point : ch_points) {
            scaled_ch_points.insert(Vec(point*scale));
        }
        return ConvexHull(scaled_ch_points, true);
    };

    ConvexHull ConvexHull::scale(const Vec& scale) const
    {
        unordered_set<Vec> scaled_ch_points;
        scaled_ch_points.reserve(size());
        for (const Vec& point : ch_points) {
            scaled_ch_points.insert(Vec(point*scale));
        }
        return ConvexHull(scaled_ch_points, true);
    };

    /**
     * Adapted from mo/pareto_front.cc
     * 
     * Cant get around just putting all the points together and pruning them together. To see why, consider convex 
     * hulls ch1={(2,0), (1,1)} and ch2={(1,1), (0,2)}. ch1 cant dominate the (1,1) from ch2 and ch2 cant dominate 
     * the (1,1) from ch1. For (1,1) to be dominated requires pruning from a set containing both (0,2) and (2,0)
    */
    ConvexHull ConvexHull::combine(const ConvexHull& other) const {
        if (other.size() > size()) {
            return other.combine(*this);
        }
        if (size() == 0) {
            return ConvexHull(other);
        } else if (other.size() == 0) {
            return ConvexHull(*this);
        }

        unordered_set<Vec> combined_points = ch_points;
        combined_points.reserve(ch_points.size() + other.ch_points.size());
        for (const Vec& point : other.ch_points) {
            combined_points.insert(point);
        }

        return ConvexHull(combined_points);
    };

    /**
     * Copied from mo/pareto_front.cc
    */
    ConvexHull ConvexHull::add(const ConvexHull& other) const 
    {
        if (ch_points.size() == 0) {
            return ConvexHull(other);
        } else if (other.ch_points.size() == 0) {
            return ConvexHull(*this);
        }

        unordered_set<Vec> summed_points;
        for (const Vec& point : ch_points) {
            for (const Vec& other_point : other.ch_points) {
                summed_points.insert(point + other_point);
            }
        }

        // constructor will prune points
        return ConvexHull(summed_points);
    };

    /**
     * Copied from mo/pareto_front.cc
    */
    ConvexHull ConvexHull::add(const Vec& v) const 
    {
        unordered_set<Vec> summed_points;
        for (const Vec& point : ch_points) {
            summed_points.insert(point + v);
        }
        return ConvexHull(summed_points, true);
    };

    ConvexHull ConvexHull::subtract(const Vec& v) const
    {
        unordered_set<Vec> subtracted_points;
        for (const Vec& point : ch_points) {
            subtracted_points.insert(point - v);
        }
        return ConvexHull(subtracted_points, true);
    };

    /**
     * Equality with another convex hull
     */
    bool ConvexHull::equals(const ConvexHull &other) const 
    {
        if (size() != other.size()) {
            return false;
        }
        for (const Vec &p : ch_points) {
            if (!other.ch_points.contains(p)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get best action for recomnmending
    */
    Vec ConvexHull::get_best_point(const Vec& context_weight, RandManager& rand_manager) const
    {
        unordered_map<Vec, double> scalarised_values;
        for (const Vec& point : ch_points) {
            scalarised_values[point] = point.dot(context_weight);
        }
        return thts::helper::get_max_key_break_ties_randomly(scalarised_values, rand_manager);
    }

    /**
     * Get maximal contextual value
     */
    double ConvexHull::get_max_linear_utility(const Vec& context_weight) const 
    {
        double max_scalarised_value = numeric_limits<double>::lowest();
        for (const Vec& point : ch_points) {
            double scalarised_value = point.dot(context_weight);
            if (scalarised_value > max_scalarised_value) {
                max_scalarised_value = scalarised_value;
            }
        }
        return max_scalarised_value;
    }

    /**
     * Hypeervolume of this convex hull
     * Using qhull library to calculate this
     * Needs the actual geometric convex hull, not the multi-objective definition of one
     * 
     * Consider points (1,2) and (2,1), with reference point (0,0)
     * Then the geometric convex hull is (0,0), (0,2), (1,2), (2,1) and (2,0) and the hypervolume is 3.5
     * 
     * This can be calculated by projecting the points onto the hyperplane defined from the reference point
     * For example, (1,2) and (2,1) would be projected to (1,0) and (2,0) onto the x-axis (hyperplane = line in 2D)
     * Similarly get (0,2) and (0,1) on the y-axis, and qhull can compute {(0,0), (0,2), (1,2), (2,1), (2,0)} is the 
     * convex hull from the points {(0,0), (1,2), (2,1), (2,0), (0,2), (0,1), (1,0)}
     * 
     * Note that because the projections are onto hyperplanes in a standard orientation, to project the points onto 
     * the hyperplane normal to the ith axis, we just set the ith coordinate to the reference point's ith coordinate
     * 
     * Note also that all projected points need to be included in further projections, otherwise the geometric convex 
     * hull may be incorrect. This is the case in the 3D example in unit tests, where the point (0,0,2) gets missed, 
     * as it requires projecting the point (1,1,2) onto (0,0,2)
     * 
     * This blows up exponentially with dimensions. So may be very slow for higher dimensions
     */
    double ConvexHull::hypervolume(const Vec& ref_point) const
    {
        // If no points, then no hypervolume
        if (ch_points.size() == 0) {
            return 0.0;
        }

        // Check that the reference point is weakly dominated by all points in the convex hull
        for (const Vec& point : ch_points) {
            if (!point.weakly_pareto_dominates(ref_point)) {
                throw runtime_error("Reference point needs to be (pareto) weakly dominated by all points in the convex "
                    "hull to compute hypervolume.");
            }
        }

        // Add initial points to the geometric hull points
        int dim = ref_point.vec.size();
        unordered_set<Vec> geometric_hull_points;

        geometric_hull_points.insert(ref_point);
        for (const Vec& point : ch_points) {
            geometric_hull_points.insert(point);
        }

        // Project the points onto the hyperplane defined at the reference point, one dimension at a time 
        for (int i = 0; i < dim; i++) {
            unordered_set<Vec> projected_points;
            for (const Vec& point : geometric_hull_points) {
                Vec projected_point = point;
                projected_point.vec[i] = ref_point.vec[i];
                projected_points.insert(projected_point);
            }
            // and remove pareto dominated the points to remove redundant projected points
            projected_points = pareto_prune(projected_points);
            geometric_hull_points.insert(projected_points.begin(), projected_points.end());
        }

        // Corner case, algorithm actually only got the reference point (worst possible value)
        // Leads to geometric_hull_points.size() == 1
        // For volume to be >0, need to have geometic_hull_points.size() > dim anyway
        // So just return 0.0 in this case
        size_t dim_size_t = static_cast<size_t>(dim);
        if (geometric_hull_points.size() <= dim_size_t) {
            return 0.0;
        }

        // Convert the geometric hull points to a flat vector for qhull
        int num_points = geometric_hull_points.size();
        std::vector<double> qhull_flat_input;
        qhull_flat_input.reserve(num_points * dim);
        for (const Vec& point : geometric_hull_points) {
            qhull_flat_input.insert(qhull_flat_input.end(), point.vec.begin(), point.vec.end());
        }

        // Run qhull to compute the convex hull, and output the hypervolume
        orgQhull::Qhull qhull;
        qhull.runQhull("", dim, num_points, qhull_flat_input.data(), "Qt Qx");
        return qhull.volume();
    }

    /**
     * Get the sparsity metric from this convex hull
     * Equation (18) in: https://arxiv.org/pdf/2103.09568
     */
    double ConvexHull::sparsity_metric() const
    {
        size_t num_points = size();
        if (num_points < 2) {
            return 0.0;
        }
        int rew_dim = reward_dim();
        vector<double> dim_j_values;
        dim_j_values.reserve(num_points);
        for (size_t i=0; i < num_points; i++) {
            dim_j_values.push_back(0.0);
        }
        double sparsity_sum = 0.0;
        for (int j=0; j < rew_dim; j++) {
            // put values for dim j into array
            size_t i=0;
            for (const Vec& v : ch_points) {
                dim_j_values[i++] = v.vec[j];
            }
            // sort array
            std::sort(dim_j_values.begin(), dim_j_values.end());
            // compute sparsity sum term for this dimension
            for (i=0; i < num_points-1; i++) {
                double diff = dim_j_values[i] - dim_j_values[i+1];
                sparsity_sum +=  diff * diff / (num_points - 1.0);
            }
        }
        return sparsity_sum;
    }

    /**
     * Minimum delta such that u+delta >= v
     */
    double min_shift_to_pareto_dominate(const Vec& u, const Vec& v) 
    {
        Vec diff = v-u;
        double min_delta = 0.0;
        for (int i=0; i<diff.dim(); i++) {
            if (diff.vec[i] > min_delta) {
                min_delta = diff.vec[i];
            }
        }
        return min_delta;
    }

    /**
     * See defn in here https://arxiv.org/pdf/2103.09568
     * For each point in convex hull
     * Find the minimum 'shift' required by some point for it to be dominated
     * The result is then the maximum over the outer loop of points
     * 
     * So inner loop = find minimum shift for another point to be able to dominate this point
     * Then we want to find the minimum shift such that all points can be dominated like that
     * Which is the maximum in the outer loop
     */
    double ConvexHull::additive_eps_metric() const 
    {
        size_t num_points = ch_points.size();
        if (num_points < 2) {
            return 0.0;
        }
        double eps = numeric_limits<double>::lowest();
        for (const Vec& v : ch_points) {
            double min_delta = numeric_limits<double>::max();
            for (const Vec& u : ch_points) {
                if (u == v) {
                    continue;
                }
                double delta = min_shift_to_pareto_dominate(u,v);
                if (delta < min_delta) {
                    min_delta = delta;
                }
            }
            if (min_delta > eps) {
                eps = min_delta;
            }
        }
        return eps;
    }

    /**
     * Pretty printing
    */
    void ConvexHull::write_to_ostream(ostream& os) const
    {
        os << "ConvexHull = {" << endl;
        for (const Vec& point : ch_points) {
            os << point << endl;
        }
        os << "}";
    }
}

/**
 * These are all copied from ParetoFront - see them for doc
*/
namespace std {
    using namespace thts;

    ConvexHull operator*(const ConvexHull& ch, double s) {
        return ch.scale(s);
    }
    
    ConvexHull operator*(double s, const ConvexHull& ch) {
        return ch.scale(s);
    }

    ConvexHull operator*(const ConvexHull& ch, const Vec& v) {
        return ch.scale(v);
    }
    ConvexHull operator*(const Vec& v, const ConvexHull& ch) {
        return ch.scale(v);
    }

    ConvexHull operator|(const ConvexHull& ch1, const ConvexHull& ch2) {
        return ch1.combine(ch2);
    }

    ConvexHull operator+(const ConvexHull& ch1, const ConvexHull& ch2) {
        return ch1.add(ch2);
    }

    ConvexHull operator+(const ConvexHull& ch, const Vec& v) {
        return ch.add(v);
    }

    ConvexHull operator+(const Vec& v, const ConvexHull& ch) {
        return ch.add(v);
    }

    ConvexHull operator-(const ConvexHull& ch, const Vec& v) {
        return ch.subtract(v);
    }

    ConvexHull operator-(const Vec& v, const ConvexHull& ch) {
        return ch.subtract(v);
    }

    bool operator==(const ConvexHull& lhs, const ConvexHull& rhs) {
        return lhs.equals(rhs);
    }

    ostream& operator<<(ostream& os, const ConvexHull& ch) {
        ch.write_to_ostream(os);
        return os;
    }
}