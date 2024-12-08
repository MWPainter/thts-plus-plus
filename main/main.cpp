#include "main/run_id.h"
#include "main/run_expr.h"
#include "main/val.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        throw runtime_error("Expecting exactly two arguments: [eval|opt] [expr_id], specifying if we want to run an "
                            "eval experiment, or perform hyperparamter optimisation.");
    }

    if (string(argv[1]) == "eval") {
        shared_ptr<vector<RunID>> run_ids = thts::get_run_ids_from_expr_id_prefix(argv[2]);
        thts::run_exprs(run_ids);
    } else if (string(argv[1]) == "opt") {  
        thts::run_hp_opt(argv[2]);
    } else if (string(argv[1]) == "noise") {
        thts::estimate_noise_for_hp_opt(argv[2]);
    } else if (string(argv[1]) == "val") {
        thts::run_valgrind_debugging(stoi(string(argv[2])));
    }

    return 0;
}

// /*
//  ********
//  * BayesOpt playing
//  ********
//  */

// #include "bayesopt/bayesopt.hpp"
// #include "bayesopt/parameters.hpp"

// #include <boost/numeric/ublas/vector.hpp>
// #include <iostream>

// class MyOptimizationC: public bayesopt::ContinuousModel
// {
//     bool two_d;
//     public:
//         MyOptimizationC(bayesopt::Parameters param, bool two_d):
//             ContinuousModel(two_d?2:4,param), two_d(two_d) 
//         {
//         };

//         double get_val(double x) {
//             if (-2.0 < x && x < 2.0) {
//                 return 5.0;
//             }
//             return x;
//         };

//         double evaluateSample(const bayesopt::vectord &query) 
//         {
//             double w = get_val(query[0] -2.0);
//             double x = get_val(query[1] -2.0);
//             if (two_d) {
//                 return 1.0*w*w + 4.0*x*x;
//             }
//             double y = get_val(query[2] -2.0);
//             double z = get_val(query[3] -2.0);
//             return 1.0*w*w + 2.0*x*x + 4.0*y*y + 8.0*z*z;
//         };
// };

// class MyOptimizationD: public bayesopt::DiscreteModel
// {
//     public:
//         MyOptimizationD(const bayesopt::vecOfvec &valid_set, bayesopt::Parameters param):
//             DiscreteModel(valid_set,param) 
//         {
//         };

//         double get_val(double x) {
//             if (-2 < x && x < 2) {
//                 return 5;
//             }
//             return x;
//         };

//         double evaluateSample(const bayesopt::vectord &query) 
//         {
//             double w = get_val(query[0] -2.0);
//             double x = get_val(query[1] -2.0);
//             return 1.0*w*w + 4.0*x*x;
//         };
// };

// int main(int argc, char* argv[]) {

//     // Dscr

//     bayesopt::Parameters par_d;
//     // par_d.l_type = L_MCMC;
//     par_d.surr_name = "sGaussianProcessNormal";
//     par_d.noise = 1e-10;
//     par_d.n_iterations = 200;
//     par_d.n_init_samples = 10;
//     par_d.n_iter_relearn = 10;
//     par_d.verbose_level = 0;
//     // par_d.verbose_level = 1;
//     // par_d.kernel.name = "kSEARD";
//     // par_d.crit_name = "cPOI";

//     bayesopt::vecOfvec valid_points;  
//     for (double i=-10.0; i<=10.0; i+=0.5) {
//         for (double j=-10.0; j<=10.0; j+=0.5) {

//             bayesopt::vectord point = bayesopt::vectord(2);
//             point[0] = i;
//             point[1] = j;
//             valid_points.push_back(point);
//         }
//     }

//     MyOptimizationD opt(valid_points, par_d);
//     bayesopt::vectord result_dscr(2);
//     opt.optimize(result_dscr);

//     cout << "Result from the (discrete) optimisation:" << result_dscr << endl;

//     // Cts - 2d
    
//     bayesopt::Parameters par_c;
//     // par_c.l_type = L_MCMC;
//     par_c.surr_name = "sGaussianProcessNormal";
//     par_c.noise = 1e-10;
//     par_c.n_iterations = 200;
//     par_c.n_init_samples = 10;
//     par_c.n_iter_relearn = 10;
//     par_c.verbose_level = 0;
//     // par_c.verbose_level = 1;
//     // par_c.kernel.name = "kSEARD";
//     // par_c.crit_name = "cPOI";

//     bayesopt::vectord lb(2);
//     lb[0] = -20.0;
//     lb[1] = -20.0;
//     bayesopt::vectord ub(2);
//     ub[0] = 20.0;
//     ub[1] = 20.0;

//     MyOptimizationC opt_c(par_c,true);
//     opt_c.setBoundingBox(lb, ub);
//     bayesopt::vectord result_cts_two(2);
//     opt_c.optimize(result_cts_two);

//     cout << "Result from the (2d) optimisation:" << result_cts_two << endl;

//     // Cts - 4d
    
//     bayesopt::Parameters par_e;
//     // par_e.l_type = L_MCMC;
//     par_e.surr_name = "sGaussianProcessNormal";
//     par_e.noise = 1e-10;
//     par_e.n_iterations = 200;
//     par_e.n_init_samples = 10;
//     par_e.n_iter_relearn = 10;
//     par_e.verbose_level = 0;
//     // par_e.verbose_level = 1;
//     // par_e.kernel.name = "kSEARD";
//     // par_e.crit_name = "cPOI";

//     bayesopt::vectord lb2(4);
//     lb2[0] = -20.0;
//     lb2[1] = -20.0;
//     lb2[2] = -20.0;
//     lb2[3] = -20.0;
//     bayesopt::vectord ub2(4);
//     ub2[0] = 20.0;
//     ub2[1] = 20.0;
//     ub2[2] = 20.0;
//     ub2[3] = 20.0;

//     MyOptimizationC opt_c2(par_e,false);
//     opt_c2.setBoundingBox(lb, ub);
//     bayesopt::vectord result(4);
//     opt_c2.optimize(result);

//     cout << "Result from the (4d) optimisation:" << result << endl;

//     return 0;
// }