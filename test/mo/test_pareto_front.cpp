#include "test/mo/test_pareto_front.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "mo/data_structures/pareto_front.h"

// includes
#include <string>
#include <utility>

#include <Eigen/Dense>


using namespace std;
using namespace thts;
using namespace thts::test;





static Eigen::ArrayXd make_vec(double a, double b) {
    Eigen::ArrayXd v(2);
    v[0] = a;
    v[1] = b;
    return v;
}

static Eigen::ArrayXd make_vec(double a, double b, double c) {
    Eigen::ArrayXd v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return v;
}

static Eigen::ArrayXd make_vec(double a, double b, double c, double d) {
    Eigen::ArrayXd v(4);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    return v;
}









/**
 * Helper to check that s1 is a subset of s2
*/
bool thts::test::set_subset(unordered_set<Vec> s1, unordered_set<Vec> s2) {
    for (const Vec& p1 : s1) {
        auto it = s2.find(p1);
        if (it == s2.end()) {
            return false;
        }
        // check that the point is the same
        if (*it != p1) {
            return false;
        }
    }
    return true;
};

/**
 * Helper to compare sets of unordered sets
*/
bool thts::test::set_equals(unordered_set<Vec> s1, unordered_set<Vec> s2) {
    return set_subset(s1,s2) && set_subset(s2,s1);
}; 








/**
 * Empty constructor
*/
TEST(Pf_Constructors, empty_constructor) {
    TestableParetoFront pf;
    EXPECT_EQ(pf.size(), 0u);
}

/**
 * Test constructing from set of tagged points, and prune fn
*/
TEST(Pf_Constructors, set_constructors) {
    unordered_set<Vec> points1 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
        make_vec(1.0,1.9),
        make_vec(1.0,1.0),
        make_vec(0.0,0.0),
    };
    TestableParetoFront pf1(points1);
    unordered_set<Vec> expected_pf1 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 2u);

    unordered_set<Vec> points2 = {
        make_vec(1.0,3.0),
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
        make_vec(3.0,1.0),
        make_vec(3.0,1.0),
    };
    TestableParetoFront pf2(points2);
    unordered_set<Vec> expected_pf2 = {
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
    };
    EXPECT_TRUE(pf2.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * Test copy constructor
*/
TEST(Pf_Constructors, copy_constructor) {
    unordered_set<Vec> points = {
        make_vec(2.0,1.0),
        make_vec(1.0,2.0),
        make_vec(1.0,1.9),
        make_vec(1.0,1.0),
        make_vec(0.0,0.0),
    };
    TestableParetoFront pf1(points);
    TestableParetoFront pf2(pf1);
    unordered_set<Vec> expected_pf = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    EXPECT_TRUE(pf2.check_fits_expected(expected_pf));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * Single argument prune is tested in the constructors
 * Testing the two argument prune funciton
*/
TEST(Pf_Arithmetic, prune) {
    TestableParetoFront pf;
    unordered_set<Vec> ref_points = {
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
    };
    unordered_set<Vec> points = {
        make_vec(2.0,3.0),
        make_vec(1.0,1.0),
        make_vec(4.0,0.5),
        make_vec(5.0,0.0),
        make_vec(-1.0,0.0),
    };
    unordered_set<Vec> expected_pruned_points = {
        make_vec(2.0,3.0),
        make_vec(4.0,0.5),
        make_vec(5.0,0.0),
    };

    unordered_set<Vec> pruned_points = pf.public_prune(ref_points, points);
    EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
} 

/**
 * Single argument prune is tested in the constructors
 * Testing the two argument prune funciton
*/
TEST(Pf_Arithmetic, prune_corner_cases) {
    TestableParetoFront pf;

    // a point in 'points' is also in 'ref_points' and should be removed
    unordered_set<Vec> ref_points = {
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
    };
    unordered_set<Vec> points = {
        make_vec(2.0,3.0),
        make_vec(1.0,1.0),
        make_vec(1.0,3.0),
        make_vec(4.0,0.5),
    };
    unordered_set<Vec> expected_pruned_points = {
        make_vec(2.0,3.0),
        make_vec(4.0,0.5),
    };

    unordered_set<Vec> pruned_points = pf.public_prune(ref_points, points);
    EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
    // 'points' contain points that dominate each other, but shouldn't be removed, because not dominated by any 
    // points in 'ref_points'
    unordered_set<Vec> ref_points2 = {
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0,0.0),
        make_vec(1.0,1.0),
        make_vec(2.0,2.0),
        make_vec(3.0,3.0),
    };
    unordered_set<Vec> expected_pruned_points2 = {
        make_vec(2.0,2.0),
        make_vec(3.0,3.0),
    };

    unordered_set<Vec> pruned_points2 = pf.public_prune(ref_points2, points2);
    EXPECT_TRUE(set_equals(pruned_points2, expected_pruned_points2));
}

/**
 * 
*/
TEST(Pf_Arithmetic, scale) {
    unordered_set<Vec> points = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    TestableParetoFront pf(points);

    // test scale
    TestableParetoFront pf1 = (TestableParetoFront) pf.scale(0.5);
    unordered_set<Vec> expected_pf1 = {
        make_vec(0.5,1.0),
        make_vec(1.0,0.5),
    };
    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 2u);

    // test operator *
    TestableParetoFront pf2 = (TestableParetoFront) (2.0 * pf);
    unordered_set<Vec> expected_pf2 = {
        make_vec(2.0,4.0),
        make_vec(4.0,2.0),
    };
    EXPECT_TRUE(pf2.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * 
*/
TEST(Pf_Arithmetic, union) {
    unordered_set<Vec> points1 = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableParetoFront pf1(points1);

    unordered_set<Vec> points2 = {
        make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };
    TestableParetoFront pf2(points2);

    unordered_set<Vec>  expected_union_pf = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };

    TestableParetoFront pf3 = (TestableParetoFront) pf1.combine(pf2);
    TestableParetoFront pf4 = (TestableParetoFront) (pf1 | pf2);

    EXPECT_TRUE(pf3.check_fits_expected(expected_union_pf));
    EXPECT_EQ(pf3.size(), 3u);

    EXPECT_TRUE(pf4.check_fits_expected(expected_union_pf));
    EXPECT_EQ(pf4.size(), 3u);
}

/**
 * Test adding
 * There is a dominated vector of (1.0,2.1) from adding 1c and 2a
 * There are two ways of making vector (2.0,2.0) from adding 1b and 2a or adding 1a and 2b
*/
TEST(Pf_Arithmetic, add_pfs) {
    unordered_set<Vec> points1 = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableParetoFront pf1(points1);

    unordered_set<Vec> points2 = {
        make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };
    TestableParetoFront pf2(points2);

    unordered_set<Vec> expected_add_pf = {
        make_vec(3.0,1.0),
        make_vec(2.0,2.0),
        make_vec(1.0,3.0),
        make_vec(0.0,3.1),
    };

    TestableParetoFront pf3 = (TestableParetoFront) pf1.add(pf2);
    TestableParetoFront pf4 = (TestableParetoFront) (pf2 + pf1);

    EXPECT_TRUE(pf3.check_fits_expected(expected_add_pf));
    EXPECT_EQ(pf3.size(), 4u);

    EXPECT_TRUE(pf4.check_fits_expected(expected_add_pf));
    EXPECT_EQ(pf4.size(), 4u);
}

/**
 * 
*/
TEST(Pf_Arithmetic, add_vector) {
    unordered_set<Vec> points = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableParetoFront pf(points);

    Vec v1 = make_vec(1.0,3.0);
    Vec v2 = make_vec(-1.0,0.0);

    unordered_set<Vec> expected_pf1 = {
        make_vec(3.0,3.0),
        make_vec(2.0,4.0),
        make_vec(1.0,4.1),
    };

    unordered_set<Vec> expected_pf2 = {
        make_vec(1.0,0.0),
        make_vec(0.0,1.0),
        make_vec(-1.0,1.1),
    };

    TestableParetoFront pf1 = (TestableParetoFront) pf.add(v1);
    TestableParetoFront pf2 = (TestableParetoFront) (pf + v2);

    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 3u);

    EXPECT_TRUE(pf2.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf2.size(), 3u);
}