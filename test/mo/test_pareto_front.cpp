#include "test/mo/test_pareto_front.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "mo/pareto_front.h"

// includes
#include <string>
#include <utility>

#include <Eigen/Dense>

// #include <qhull>
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/RboxPoints.h"


using namespace std;
using namespace thts;
using namespace thts::test;
using namespace orgQhull;





Eigen::ArrayXd make_vec(double a, double b) {
    Eigen::ArrayXd v(2);
    v[0] = a;
    v[1] = b;
    return v;
}

Eigen::ArrayXd make_vec(double a, double b, double c) {
    Eigen::ArrayXd v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return v;
}

Eigen::ArrayXd make_vec(double a, double b, double c, double d) {
    Eigen::ArrayXd v(4);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    return v;
}





TEST(CH, CH1) {    
    RboxPoints rbox;
    rbox.appendPoints("100");
    Qhull qhull;
    qhull.runQhull(rbox, "");
    // QhullFacetList facets(qhull);
    cout << qhull.area() << endl;
    // cout << qhull.points() << endl;
    cout << qhull.points().size() << endl;
    cout << qhull.volume() << endl;
}

TEST(CH, CH2) {
    std::vector<std::vector<double>> points{
        {-6, 0, 0, 0, 0},
        {-3, 0, -2, 0, 0},
        {-1, 0, -2, 0, 0},
        {0, -6, 0, 0, 0},
        {0, -3, -2, 0, 0},
        {0, -1, -4, 0, 0},
        {0, -1, -2, -2, 0},
        {0, -1, -2, 0, -1},
        {0, -1, -2, 0, 0},
        {0, -1, 0, -2, -1},
        {0, 0, -6, 0, 0},
        {0, 0, -1, -1, -3},
        {0, 0, 0, -6, 0},
        {0, 0, 0, 0, -6},
    };

    // compute number of dimensions
    const auto dimensions = std::begin(points)->size();

    // compile input for qhull
    std::vector<double> flat_input;
    for (const auto &p : points) {
        flat_input.insert(std::end(flat_input), std::begin(p), std::end(p));
    }

    // compute convex hull
    orgQhull::Qhull qhull;
    qhull.runQhull("", dimensions, points.size(), flat_input.data(), "Qt Qx");
    std::set<std::vector<double>> convex_hull;

    for (const auto &facet : qhull.facetList()) {
        for (const auto &vertex : facet.vertices()) {
            double *coordinates = vertex.point().coordinates();
            std::vector<double> p(coordinates, coordinates + dimensions);
            convex_hull.insert(p);
        }
    }

    std::cout << convex_hull.size() << '\n';
} 

TEST(CH, CH3) {
    std::vector<std::vector<double>> points{
        {-3, -3},
        {-3, 3},
        {3, -3},
        {3, 2},
        {2,3},
        {0,0},
        {2,1},
        {-1,0.5},
        {-0.5,1},
        {2.5,2.5},
    };

    // compute number of dimensions
    const auto dimensions = std::begin(points)->size();

    // compile input for qhull
    std::vector<double> flat_input;
    for (const auto &p : points) {
        flat_input.insert(std::end(flat_input), std::begin(p), std::end(p));
    }

    // compute convex hull
    orgQhull::Qhull qhull;
    qhull.runQhull("", dimensions, points.size(), flat_input.data(), "Qt Qx");
    std::set<std::vector<double>> convex_hull;

    for (const auto &facet : qhull.facetList()) {
        for (const auto &vertex : facet.vertices()) {
            double *coordinates = vertex.point().coordinates();
            std::vector<double> p(coordinates, coordinates + dimensions);
            convex_hull.insert(p);
        }
    }

    std::cout << convex_hull.size() << " expect 5" << endl;
    cout << qhull.area() << endl;
    cout << qhull.volume() << endl;
    for (const vector<double>& vec : convex_hull) {
        for (double d : vec) {
            cout << d << " ";
        }
        cout << endl;
    }
} 


/**
 * Tests that equality works for Tagged points
*/
TEST(Pf_TaggedPoint, equality_one_dim) {
    TaggedPoint<int> p1(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p2(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p3(make_vec(1.0,2.0,3.0), 2);
    TaggedPoint<int> p4(make_vec(1.0,2.0,0.0), 1);
    TaggedPoint<int> p5(make_vec(0.0,0.0,0.0), 1);

    EXPECT_TRUE(p1.equals(p2));
    EXPECT_TRUE(p1.equals(p3)); // different tag, but equality should ignore this
    EXPECT_FALSE(p1.equals(p4));
    EXPECT_FALSE(p1.equals(p5));

    // check operator== too
    EXPECT_TRUE(p1 == p2);
    EXPECT_TRUE(p1 == p3);
    EXPECT_FALSE(p1 == p4);
    EXPECT_FALSE(p1 == p5);

    // also check hashes
    EXPECT_EQ(p1.hash(), p2.hash());
    EXPECT_EQ(p1.hash(), p3.hash());
    EXPECT_NE(p1.hash(), p4.hash());
    EXPECT_NE(p1.hash(), p5.hash());
} 

/**
 * Tests that equality throws errors for Tagged points with different dims
*/
TEST(Pf_TaggedPoint, equality_diff_dims) {
    TaggedPoint<int> p1(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p2(make_vec(1.0,2.0), 1);
    TaggedPoint<int> p3(make_vec(1.0,2.0,3.0,4.0), 1);

    EXPECT_ANY_THROW(p1.equals(p2));
    EXPECT_ANY_THROW(p1.equals(p3));
    EXPECT_NE(p1.hash(), p2.hash());
    EXPECT_NE(p1.hash(), p3.hash());
}

/**
 * Test copy constructor makes a copy
*/
TEST(Pf_TaggedPoint, copy_constructor) {
    TaggedPoint<int> p1(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p2(p1);

    EXPECT_TRUE(p1.equals(p2));
    EXPECT_TRUE(p1 == p2);
    EXPECT_EQ(p1.hash(), p2.hash());

    p2.point = make_vec(0.0,0.0,0.0);

    EXPECT_FALSE(p1.equals(p2));
    EXPECT_FALSE(p1 == p2);
    EXPECT_NE(p1.hash(), p2.hash());
}

/**
 * Test weakly pareto domination
*/
TEST(Pf_TaggedPoint, weak_pareto_domination) {
    TaggedPoint<int> p1(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p2(make_vec(1.0,2.0,3.0), 2);
    TaggedPoint<int> p3(make_vec(1.1,2.0,3.0), 3);
    TaggedPoint<int> p4(make_vec(1.0,2.1,3.0), 4);
    TaggedPoint<int> p5(make_vec(1.0,2.0,3.1), 5);
    TaggedPoint<int> p6(make_vec(4.0,4.0,4.0), 6);

    EXPECT_TRUE(p1.weakly_pareto_dominates(p1));
    EXPECT_TRUE(p1.weakly_pareto_dominates(p2));
    EXPECT_TRUE(p2.weakly_pareto_dominates(p1));
    EXPECT_FALSE(p1.weakly_pareto_dominates(p3));
    EXPECT_TRUE(p3.weakly_pareto_dominates(p1));
    EXPECT_FALSE(p1.weakly_pareto_dominates(p4));
    EXPECT_TRUE(p4.weakly_pareto_dominates(p1));
    EXPECT_FALSE(p1.weakly_pareto_dominates(p5));
    EXPECT_TRUE(p5.weakly_pareto_dominates(p1));
    EXPECT_FALSE(p1.weakly_pareto_dominates(p6));
    EXPECT_TRUE(p6.weakly_pareto_dominates(p1));
}

/**
 * Test weakly pareto domination throws error if comparing different dim points
*/
TEST(Pf_TaggedPoint, weak_pareto_domination_errors) {
    TaggedPoint<int> p1(make_vec(1.0,2.0,3.0), 1);
    TaggedPoint<int> p2(make_vec(1.0,2.0), 1);
    TaggedPoint<int> p3(make_vec(1.0,2.0,3.0,4.0), 1);

    EXPECT_ANY_THROW(p1.weakly_pareto_dominates(p2));
    EXPECT_ANY_THROW(p1.weakly_pareto_dominates(p3));
    EXPECT_ANY_THROW(p2.weakly_pareto_dominates(p1));
    EXPECT_ANY_THROW(p3.weakly_pareto_dominates(p1));
}







/**
 * Empty constructor
*/
TEST(Pf_Constructors, empty_constructor) {
    TestableParetoFront<int> pf;
    EXPECT_EQ(pf.size(), 0u);
}

/**
 * Test constructing from set constructions, and prune fn
*/
TEST(Pf_Constructors, vector_constructors) {
    vector<pair<Eigen::ArrayXd,string>> points1 = {
        make_pair(make_vec(1.0,2.0), "1"),
        make_pair(make_vec(2.0,1.0), "2"),
        make_pair(make_vec(1.0,1.9), "1"),
        make_pair(make_vec(1.0,1.0), "1"),
        make_pair(make_vec(0.0,0.0), "1"), 
    };
    TestableParetoFront<string> pf1(points1);
    unordered_set<TaggedPoint<string>> expected_pf1 = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "2"),
    };
    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 2u);
    
    vector<pair<Eigen::ArrayXd,string>> points2 = {
        make_pair(make_vec(1.0,3.0), "1"),
        make_pair(make_vec(1.0,3.0), "1"),
        make_pair(make_vec(3.0,1.0), "3"),
        make_pair(make_vec(3.0,1.0), "1"),
        make_pair(make_vec(3.0,1.0), "1"),
    };
    TestableParetoFront<string> pf2(points2);
    unordered_set<TaggedPoint<unordered_set<string>>> expected_pf2 = {
        TaggedPoint<unordered_set<string>>(make_vec(1.0,3.0), {"1"}),
        TaggedPoint<unordered_set<string>>(make_vec(3.0,1.0), {"1", "3"}),
    };
    EXPECT_TRUE(pf2.check_fits_expected_multitag(expected_pf2));
    EXPECT_EQ(pf2.size(), 2u);

    // vector<Eigen::ArrayXd> points3 = {
    //     make_vec(1.0,2.0),
    //     make_vec(2.0,1.0),
    //     make_vec(1.0,1.9),
    //     make_vec(1.0,1.0),
    //     make_vec(0.0,0.0),
    // };
    // TestableParetoFront<string> pf3(points3, "1");
    // unordered_set<TaggedPoint<string>> expected_pf3 = {
    //     TaggedPoint<string>(make_vec(1.0,2.0), "1"),
    //     TaggedPoint<string>(make_vec(2.0,1.0), "1"),
    // };
    // EXPECT_TRUE(pf3.check_fits_expected(expected_pf3));
    // EXPECT_EQ(pf3.size(), 2u);
    
    // vector<Eigen::ArrayXd> points4 = {
    //     make_vec(1.0,3.0),
    //     make_vec(1.0,3.0),
    //     make_vec(3.0,1.0),
    //     make_vec(3.0,1.0),
    //     make_vec(3.0,1.0),
    // };
    // TestableParetoFront<string> pf4(points4, "1");
    // unordered_set<TaggedPoint<string>> expected_pf4 = {
    //     TaggedPoint<string>(make_vec(1.0,3.0), "1"),
    //     TaggedPoint<string>(make_vec(3.0,1.0), "1"),
    // };
    // EXPECT_TRUE(pf4.check_fits_expected(expected_pf4));
    // EXPECT_EQ(pf4.size(), 2u);
}

/**
 * Test constructing from set of tagged points, and prune fn
*/
TEST(Pf_Constructors, set_constructors) {
    unordered_set<TaggedPoint<string>> points1 = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "1"),
        TaggedPoint<string>(make_vec(1.0,1.9), "1"),
        TaggedPoint<string>(make_vec(1.0,1.0), "1"),
        TaggedPoint<string>(make_vec(0.0,0.0), "1"),
    };
    TestableParetoFront<string> pf1(points1);
    unordered_set<TaggedPoint<string>> expected_pf1 = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "1"),
    };
    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 2u);
    
    unordered_set<TaggedPoint<string>> points2 = {
        TaggedPoint<string>(make_vec(1.0,3.0), "1"),
        TaggedPoint<string>(make_vec(1.0,3.0), "1"),
        TaggedPoint<string>(make_vec(3.0,1.0), "3"),
        TaggedPoint<string>(make_vec(3.0,1.0), "1"),
        TaggedPoint<string>(make_vec(3.0,1.0), "1"),
    };
    TestableParetoFront<string> pf2(points2);
    unordered_set<TaggedPoint<unordered_set<string>>> expected_pf2 = {
        TaggedPoint<unordered_set<string>>(make_vec(1.0,3.0), {"1"}),
        TaggedPoint<unordered_set<string>>(make_vec(3.0,1.0), {"1","3"}),
    };
    EXPECT_TRUE(pf2.check_fits_expected_multitag(expected_pf2));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * Test copy constructor
*/
TEST(Pf_Constructors, copy_constructor) {
    unordered_set<TaggedPoint<string>> points = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "1"),
        TaggedPoint<string>(make_vec(1.0,1.9), "1"),
        TaggedPoint<string>(make_vec(1.0,1.0), "1"),
        TaggedPoint<string>(make_vec(0.0,0.0), "1"),
    };
    TestableParetoFront<string> pf1(points);
    TestableParetoFront<string> pf2(pf1);
    unordered_set<TaggedPoint<string>> expected_pf = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "1"),
    };
    EXPECT_TRUE(pf2.check_fits_expected(expected_pf));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * Tests 'set_tags'
*/
TEST(Pf_Constructors, setting_tags) {
    vector<pair<Eigen::ArrayXd,string>> points = {
        make_pair(make_vec(1.0,2.0), "1"),
        make_pair(make_vec(2.0,1.0), "2"),
    };
    TestableParetoFront<string> pf(points);
    unordered_set<TaggedPoint<string>> expected_pf1 = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "2"),
    };
    EXPECT_TRUE(pf.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf.size(), 2u);

    pf.set_tags("3");
    unordered_set<TaggedPoint<string>> expected_pf2 = {
        TaggedPoint<string>(make_vec(1.0,2.0), "3"),
        TaggedPoint<string>(make_vec(2.0,1.0), "3"),
    };
    EXPECT_TRUE(pf.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf.size(), 2u);

}

/**
 * Single argument prune is tested in the constructors
 * Testing the two argument prune funciton
*/
TEST(Pf_Arithmetic, prune) {
    TestableParetoFront<int> pf;
    unordered_set<TaggedPoint<int>> ref_points = {
        TaggedPoint<int>(make_vec(1.0,3.0), 1),
        TaggedPoint<int>(make_vec(3.0,1.0), 1),
    };
    unordered_set<TaggedPoint<int>> points = {
        TaggedPoint<int>(make_vec(2.0,3.0), 1),
        TaggedPoint<int>(make_vec(1.0,1.0), 1),
        TaggedPoint<int>(make_vec(4.0,0.5), 1),
        TaggedPoint<int>(make_vec(5.0,0.0), 1),
        TaggedPoint<int>(make_vec(-1.0,0.0), 1),
    };
    unordered_set<TaggedPoint<int>> expected_pruned_points = {
        TaggedPoint<int>(make_vec(2.0,3.0), 1),
        TaggedPoint<int>(make_vec(4.0,0.5), 1),
        TaggedPoint<int>(make_vec(5.0,0.0), 1),
    };

    unordered_set<TaggedPoint<int>> pruned_points = pf.public_prune(ref_points, points);
    EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
} 

/**
 * Single argument prune is tested in the constructors
 * Testing the two argument prune funciton
*/
TEST(Pf_Arithmetic, prune_corner_cases) {
    TestableParetoFront<int> pf;

    // a point in 'points' is also in 'ref_points' and should be removed
    unordered_set<TaggedPoint<int>> ref_points = {
        TaggedPoint<int>(make_vec(1.0,3.0), 1),
        TaggedPoint<int>(make_vec(3.0,1.0), 1),
    };
    unordered_set<TaggedPoint<int>> points = {
        TaggedPoint<int>(make_vec(2.0,3.0), 1),
        TaggedPoint<int>(make_vec(1.0,1.0), 1),
        TaggedPoint<int>(make_vec(1.0,3.0), 1),
        TaggedPoint<int>(make_vec(4.0,0.5), 1),
    };
    unordered_set<TaggedPoint<int>> expected_pruned_points = {
        TaggedPoint<int>(make_vec(2.0,3.0), 1),
        TaggedPoint<int>(make_vec(4.0,0.5), 1),
    };

    unordered_set<TaggedPoint<int>> pruned_points = pf.public_prune(ref_points, points);
    EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
    // 'points' contain points that dominate each other, but shouldn't be removed, because not dominated by any 
    // points in 'ref_points'
    unordered_set<TaggedPoint<int>> ref_points2 = {
        TaggedPoint<int>(make_vec(1.0,3.0), 1),
        TaggedPoint<int>(make_vec(3.0,1.0), 1),
    };
    unordered_set<TaggedPoint<int>> points2 = {
        TaggedPoint<int>(make_vec(0.0,0.0), 1),
        TaggedPoint<int>(make_vec(1.0,1.0), 1),
        TaggedPoint<int>(make_vec(2.0,2.0), 1),
        TaggedPoint<int>(make_vec(3.0,3.0), 1),
    };
    unordered_set<TaggedPoint<int>> expected_pruned_points2 = {
        TaggedPoint<int>(make_vec(2.0,2.0), 1),
        TaggedPoint<int>(make_vec(3.0,3.0), 1),
    };

    unordered_set<TaggedPoint<int>> pruned_points2 = pf.public_prune(ref_points2, points2);
    EXPECT_TRUE(set_equals(pruned_points2, expected_pruned_points2));
}

/**
 * 
*/
TEST(Pf_Arithmetic, scale) {
    unordered_set<TaggedPoint<string>> points = {
        TaggedPoint<string>(make_vec(1.0,2.0), "1"),
        TaggedPoint<string>(make_vec(2.0,1.0), "1"),
    };
    TestableParetoFront<string> pf(points);

    // test scale
    TestableParetoFront<string> pf1 = (TestableParetoFront<string>) pf.scale(0.5);
    unordered_set<TaggedPoint<string>> expected_pf1 = {
        TaggedPoint<string>(make_vec(0.5,1.0), "1"),
        TaggedPoint<string>(make_vec(1.0,0.5), "1"),
    };
    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 2u);

    // test operator *
    TestableParetoFront<string> pf2 = (TestableParetoFront<string>) (pf * 2.0);
    unordered_set<TaggedPoint<string>> expected_pf2 = {
        TaggedPoint<string>(make_vec(2.0,4.0), "1"),
        TaggedPoint<string>(make_vec(4.0,2.0), "1"),
    };
    EXPECT_TRUE(pf2.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf2.size(), 2u);
}

/**
 * 
*/
TEST(Pf_Arithmetic, union) {
    unordered_set<TaggedPoint<string>> points1 = {
        TaggedPoint<string>(make_vec(2.0,0.0), "1a"),
        TaggedPoint<string>(make_vec(1.0,1.0), "1b"),
        TaggedPoint<string>(make_vec(0.0,1.1), "1c"),
    };
    TestableParetoFront<string> pf1(points1);

    unordered_set<TaggedPoint<string>> points2 = {
        TaggedPoint<string>(make_vec(1.0,1.0), "2a"),
        TaggedPoint<string>(make_vec(0.0,2.0), "2b"),
    };
    TestableParetoFront<string> pf2(points2);

    unordered_set<TaggedPoint<unordered_set<string>>> expected_union_pf = {
        TaggedPoint<unordered_set<string>>(make_vec(2.0,0.0), {"1a"}),
        TaggedPoint<unordered_set<string>>(make_vec(1.0,1.0), {"1b","2a"}),
        TaggedPoint<unordered_set<string>>(make_vec(0.0,2.0), {"2b"}),
    };

    TestableParetoFront<string> pf3 = (TestableParetoFront<string>) pf1.combine(pf2);
    TestableParetoFront<string> pf4 = (TestableParetoFront<string>) (pf1 % pf2);

    EXPECT_TRUE(pf3.check_fits_expected_multitag(expected_union_pf));
    EXPECT_EQ(pf3.size(), 3u);

    EXPECT_TRUE(pf4.check_fits_expected_multitag(expected_union_pf));
    EXPECT_EQ(pf4.size(), 3u);
}

/**
 * Test adding
 * There is a dominated vector of (1.0,2.1) from adding 1c and 2a
 * There are two ways of making vector (2.0,2.0) from adding 1b and 2a or adding 1a and 2b
*/
TEST(Pf_Arithmetic, add_pfs) {
    unordered_set<TaggedPoint<string>> points1 = {
        TaggedPoint<string>(make_vec(2.0,0.0), "1a"),
        TaggedPoint<string>(make_vec(1.0,1.0), "1b"),
        TaggedPoint<string>(make_vec(0.0,1.1), "1c"),
    };
    TestableParetoFront<string> pf1(points1);

    unordered_set<TaggedPoint<string>> points2 = {
        TaggedPoint<string>(make_vec(1.0,1.0), "2a"),
        TaggedPoint<string>(make_vec(0.0,2.0), "2b"),
    };
    TestableParetoFront<string> pf2(points2);

    unordered_set<TaggedPoint<unordered_set<string>>> expected_add_pf = {
        TaggedPoint<unordered_set<string>>(make_vec(3.0,1.0), {"1a","2a"}),
        TaggedPoint<unordered_set<string>>(make_vec(2.0,2.0), {"1a","1b","2a","2b"}),
        TaggedPoint<unordered_set<string>>(make_vec(1.0,3.0), {"1b","2b"}),
        TaggedPoint<unordered_set<string>>(make_vec(0.0,3.1), {"1c","2b"}),
    };

    TestableParetoFront<string> pf3 = (TestableParetoFront<string>) pf1.add(pf2);
    TestableParetoFront<string> pf4 = (TestableParetoFront<string>) (pf2 + pf1);

    EXPECT_TRUE(pf3.check_fits_expected_multitag(expected_add_pf));
    EXPECT_EQ(pf3.size(), 4u);

    EXPECT_TRUE(pf4.check_fits_expected_multitag(expected_add_pf));
    EXPECT_EQ(pf4.size(), 4u);
}

/**
 * 
*/
TEST(Pf_Arithmetic, add_vector) {
    unordered_set<TaggedPoint<string>> points = {
        TaggedPoint<string>(make_vec(2.0,0.0), "1a"),
        TaggedPoint<string>(make_vec(1.0,1.0), "1b"),
        TaggedPoint<string>(make_vec(0.0,1.1), "1c"),
    };
    TestableParetoFront<string> pf(points);

    Eigen::ArrayXd v1 = make_vec(1.0,3.0);
    Eigen::ArrayXd v2 = make_vec(-1.0,0.0);

    unordered_set<TaggedPoint<string>> expected_pf1 = {
        TaggedPoint<string>(make_vec(3.0,3.0), "1a"),
        TaggedPoint<string>(make_vec(2.0,4.0), "1b"),
        TaggedPoint<string>(make_vec(1.0,4.1), "1c"),
    };

    unordered_set<TaggedPoint<string>> expected_pf2 = {
        TaggedPoint<string>(make_vec(1.0,0.0), "1a"),
        TaggedPoint<string>(make_vec(0.0,1.0), "1b"),
        TaggedPoint<string>(make_vec(-1.0,1.1), "1c"),
    };

    TestableParetoFront<string> pf1 = (TestableParetoFront<string>) pf.add(v1);
    TestableParetoFront<string> pf2 = (TestableParetoFront<string>) (pf + v2);

    EXPECT_TRUE(pf1.check_fits_expected(expected_pf1));
    EXPECT_EQ(pf1.size(), 3u);

    EXPECT_TRUE(pf2.check_fits_expected(expected_pf2));
    EXPECT_EQ(pf2.size(), 3u);
}