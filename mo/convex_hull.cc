#include "mo/convex_hull.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <limits>

#include <iostream>
#include <stdexcept>

// #include "lemon/lp.h"
#include "coin-or/ClpSimplex.hpp"
#include "coin-or/CoinPackedMatrix.hpp"



/**
 * ConvexHull implementation
 */
namespace thts {
    using namespace std;
    // using namespace lemon;
    
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
     * Constructor initialising from a heuristic val
    */
    template <typename T>
    ConvexHull<T>::ConvexHull(const Eigen::ArrayXd& heuristic_val, const T& tag) :
        ch_points()
    {
        ch_points.insert(TaggedPoint<T>(heuristic_val, tag));
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
        ch_points(std::move(ch.ch_points)) 
    {
    };

    /**
     * Copy assignment
    */
    template <typename T>
    ConvexHull<T>& ConvexHull<T>::operator=(const ConvexHull<T>& ch) 
    {
        this->ch_points = ch.ch_points;
        return *this;
    }

    /**
     * Move assignment
    */
    template <typename T>
    ConvexHull<T>& ConvexHull<T>::operator=(ConvexHull<T>&& ch) 
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
     * But a TODO could be to check this out? Any maybe use std::shared_ptr<std::unordered_set<TaggedPoint<T>>> for 
     * ch_points
     * 
    */
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator*=(double rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs * rhs;
        return lhs;
    }

    /**
     * |=
    */
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator|=(ConvexHull<T>& rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs | rhs;
        return lhs;
    }
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator|=(ConvexHull<T>&& rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs | rhs;
        return lhs;
    }

    /**
     * +=
    */
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator+=(ConvexHull<T>& rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs + rhs;
        return lhs;
    }
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator+=(ConvexHull<T>&& rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs + rhs;
        return lhs;
    }
    template <typename T>
    inline ConvexHull<T>& ConvexHull<T>::operator+=(Eigen::ArrayXd& rhs)
    {
        ConvexHull<T>& lhs = *this;
        lhs = lhs + rhs;
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
     */
    template <typename T>
    bool ConvexHull<T>::strongly_convex_dominated(
        const unordered_set<TaggedPoint<T>>& ref_points, 
        const TaggedPoint<T>& point) const
    {   
        // Base case where lp will be unbounded and would throw an error
        if (ref_points.size() == 0 || (ref_points.size() == 1 && ref_points.contains(point))) {
            return false;
        }

        // Size of lp
        int dim = point.point.size();
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
        for (const TaggedPoint<T>& ref_p : ref_points) {
            if (ref_p == point) continue;
            Eigen::ArrayXd diff = point.point - ref_p.point; // p-p_k
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

        // For debugging
        // cout << "In lp solver:" << endl;
        // cout << "Obj values = " << lp.objectiveValue() << endl;
        // const double *soltn = lp.primalColumnSolution();
        // for (int i=0; i<num_vars; i++) {
        //     cout << "Var[" << i << "] = " << soltn[i] << endl;
        // }
        // cout << endl;

        // Also seems like should use lp.numberPrimalInfeasibilities() to check actually got a solution
        if (lp.numberPrimalInfeasibilities() > 0) {
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
    // bool ConvexHull<T>::strongly_convex_dominated(
    //     const unordered_set<TaggedPoint<T>>& ref_points, 
    //     const TaggedPoint<T>& point) const
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
    //     for (const TaggedPoint<T>& ref_p : ref_points) {
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
     * Prunes a set of 'points' to a set of points that form a Convex Hull
     * 
     * Because working with a single set of points, 'pruned_points' will always contain *it in the 
     * strongly_convex_dominated call, so set 'ignore_if_point_in_ref_points' to true here, to avoid all points being 
     * pruned by themselves
     * 
     * 
    */
    template <typename T>
    unordered_set<TaggedPoint<T>> ConvexHull<T>::prune(const unordered_set<TaggedPoint<T>>& points) const 
    {
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
        for (const TaggedPoint<T>& point : ch_points) {
            point.tag = new_tag;
        }
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
     * Adapted from mo/pareto_front.cc
     * 
     * Cant get around just putting all the points together and pruning them together. To see why, consider convex 
     * hulls ch1={(2,0), (1,1)} and ch2={(1,1), (0,2)}. ch1 cant dominate the (1,1) from ch2 and ch2 cant dominate 
     * the (1,1) from ch1. For (1,1) to be dominated requires pruning from a set containing both (0,2) and (2,0)
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

        unordered_set<TaggedPoint<T>> combined_points = ch_points;
        combined_points.reserve(ch_points.size() + other.ch_points.size());
        for (const TaggedPoint<T>& point : other.ch_points) {
            combined_points.insert(point);
        }

        return ConvexHull<T>(combined_points);
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

    /**
     * Equality with another convex hull
     */
    template<typename T>
    bool ConvexHull<T>::equals(const ConvexHull<T> &other) const 
    {
        if (size() != other.size()) {
            return false;
        }
        for (const TaggedPoint<T> &p : ch_points) {
            if (!other.ch_points.contains(p)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get best action for recomnmending
    */
    template <typename T>
    TaggedPoint<T> ConvexHull<T>::get_best_point(Eigen::ArrayXd& context_weight, RandManager& rand_manager) const
    {
        unordered_map<TaggedPoint<T>, double> scalarised_values;
        for (const TaggedPoint<T>& tagged_point : ch_points) {
            scalarised_values[tagged_point] = thts::helper::dot(tagged_point.point, context_weight);
        }
        return thts::helper::get_max_key_break_ties_randomly(scalarised_values, rand_manager);
    }

    template <typename T>
    T ConvexHull<T>::get_best_point_tag(Eigen::ArrayXd& context_weight, RandManager& rand_manager)  const
    {
        return get_best_point(context_weight,rand_manager).tag;
    }

    /**
     * Get maximal contextual value
     */
    template <typename T>
    double ConvexHull<T>::get_max_linear_utility(Eigen::ArrayXd& context_weight) const 
    {
        double max_scalarised_value = numeric_limits<double>::lowest();
        for (const TaggedPoint<T>& tagged_point : ch_points) {
            double scalarised_value = thts::helper::dot(tagged_point.point, context_weight);
            if (scalarised_value > max_scalarised_value) {
                max_scalarised_value = scalarised_value;
            }
        }
        return max_scalarised_value;
    }

    /**
     * Pretty printing
    */
    template <typename T>
    void ConvexHull<T>::write_to_ostream(ostream& os) const
    {
        os << "ConvexHull = {" << endl;
        for (const TaggedPoint<T>& point : ch_points) {
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
    bool operator==(const ConvexHull<T>& lhs, const ConvexHull<T>& rhs) {
        return lhs.equals(rhs);
    }

    template <typename T>
    ostream& operator<<(ostream& os, const ConvexHull<T>& ch) {
        ch.write_to_ostream(os);
        return os;
    }
}