#include "test/mo/test_convex_hull.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "mo/data_structures/convex_hull.h"

// includes
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





/**
 * Used to play around with/learn qhull 
 */
TEST(ch_qhull, test1) {    
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

/**
 * Used to play around with/learn qhull  
 */
TEST(ch_qhull, test2) {    
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

/**
 * Used to play around with/learn qhull 
 */
TEST(ch_qhull, test3) {    
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
 * Empty constructor
*/
TEST(Ch_Constructors, empty_constructor) {
    TestableConvexHull ch;
    EXPECT_EQ(ch.size(), 0u);
}

/**
 * Test constructing from set constructions, and prune fn
*/
// TEST(Ch_Constructors, vector_constructors) {
//     vector<pair<Eigen::ArrayXd,string>> points1 = {
//         make_pair(make_vec(1.0,2.0), "1"),
//         make_pair(make_vec(2.0,1.0), "2"),
//         make_pair(make_vec(1.0,1.9), "1"),
//         make_pair(make_vec(1.0,1.0), "1"),
//         make_pair(make_vec(0.0,0.0), "1"), 
//     };
//     TestableConvexHull ch1(points1);
//     unordered_set<TaggedPoint> expected_ch1 = {
//         TaggedPoint(make_vec(1.0,2.0), "1"),
//         TaggedPoint(make_vec(2.0,1.0), "2"),
//     };
//     EXPECT_TRUE(ch1.check_fits_expected(expected_ch1));
//     EXPECT_EQ(ch1.size(), 2u);
    
//     vector<pair<Eigen::ArrayXd,string>> points2 = {
//         make_pair(make_vec(1.0,3.0), "1"),
//         make_pair(make_vec(1.0,3.0), "1"),
//         make_pair(make_vec(3.0,1.0), "3"),
//         make_pair(make_vec(3.0,1.0), "1"),
//         make_pair(make_vec(3.0,1.0), "1"),
//     };
//     TestableConvexHull ch2(points2);
//     unordered_set<TaggedPoint<unordered_set>> expected_ch2 = {
//         TaggedPoint<unordered_set>(make_vec(1.0,3.0), {"1"}),
//         TaggedPoint<unordered_set>(make_vec(3.0,1.0), {"1", "3"}),
//     };
//     EXPECT_TRUE(ch2.check_fits_expected(expected_ch2));
//     EXPECT_EQ(ch2.size(), 2u);

//     // vector<Eigen::ArrayXd> points3 = {
//     //     make_vec(1.0,2.0),
//     //     make_vec(2.0,1.0),
//     //     make_vec(1.0,1.9),
//     //     make_vec(1.0,1.0),
//     //     make_vec(0.0,0.0),
//     // };
//     // TestableConvexHull ch3(points3, "1");
//     // unordered_set<TaggedPoint> expected_ch3 = {
//     //     TaggedPoint(make_vec(1.0,2.0), "1"),
//     //     TaggedPoint(make_vec(2.0,1.0), "1"),
//     // };
//     // EXPECT_TRUE(ch3.check_fits_expected(expected_ch3));
//     // EXPECT_EQ(ch3.size(), 2u);
    
//     // vector<Eigen::ArrayXd> points4 = {
//     //     make_vec(1.0,3.0),
//     //     make_vec(1.0,3.0),
//     //     make_vec(3.0,1.0),
//     //     make_vec(3.0,1.0),
//     //     make_vec(3.0,1.0),
//     // };
//     // TestableConvexHull ch4(points4, "1");
//     // unordered_set<TaggedPoint> expected_ch4 = {
//     //     TaggedPoint(make_vec(1.0,3.0), "1"),
//     //     TaggedPoint(make_vec(3.0,1.0), "1"),
//     // };
//     // EXPECT_TRUE(ch4.check_fits_expected(expected_ch4));
//     // EXPECT_EQ(ch4.size(), 2u);
// }

/**
 * Test constructing from set of tagged points, and prune fn
*/
TEST(Ch_Constructors, set_constructors) {
    unordered_set<Vec> points1 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
        make_vec(1.0,1.9),
        make_vec(1.0,1.0),
        make_vec(0.0,0.0),
    };
    TestableConvexHull ch1(points1);
    unordered_set<Vec> expected_ch1 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    EXPECT_TRUE(ch1.check_fits_expected(expected_ch1));
    EXPECT_EQ(ch1.size(), 2u);

    unordered_set<Vec> points2 = {
        make_vec(1.0,3.0),
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
        make_vec(3.0,1.0),
        make_vec(3.0,1.0),
    };
    TestableConvexHull ch2(points2);
    unordered_set<Vec> expected_ch2 = {
        make_vec(1.0,3.0),
        make_vec(3.0,1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected_ch2));
    EXPECT_EQ(ch2.size(), 2u);
}

/**
 * Test copy constructor
*/
TEST(Ch_Constructors, copy_constructor) {
    unordered_set<Vec> points = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
        make_vec(1.0,1.9),
        make_vec(1.0,1.0),
        make_vec(0.0,0.0),
    };
    TestableConvexHull ch1(points);
    TestableConvexHull ch2(ch1);
    unordered_set<Vec> expected_ch = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected_ch));
    EXPECT_EQ(ch2.size(), 2u);
}

/**
 * Tests 'set_tags'
*/
// TEST(Ch_Constructors, setting_tags) {
//     vector<pair<Eigen::ArrayXd,string>> points = {
//         make_pair(make_vec(1.0,2.0), "1"),
//         make_pair(make_vec(2.0,1.0), "2"),
//     };
//     TestableConvexHull ch(points);
//     unordered_set<TaggedPoint> expected_ch1 = {
//         TaggedPoint(make_vec(1.0,2.0), "1"),
//         TaggedPoint(make_vec(2.0,1.0), "2"),
//     };
//     EXPECT_TRUE(ch.check_fits_expected(expected_ch1));
//     EXPECT_EQ(ch.size(), 2u);

//     ch.set_tags("3");
//     unordered_set<TaggedPoint> expected_ch2 = {
//         TaggedPoint(make_vec(1.0,2.0), "3"),
//         TaggedPoint(make_vec(2.0,1.0), "3"),
//     };
//     EXPECT_TRUE(ch.check_fits_expected(expected_ch2));
//     EXPECT_EQ(ch.size(), 2u);

// }

/**
 * Tests 'equals' relationship
 */
TEST(Ch_Arithmetic, equality) {
    vector<ConvexHull> chs;

    unordered_set<Vec> points1 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    chs.push_back(ConvexHull(points1));

    unordered_set<Vec> points2 = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    chs.push_back(ConvexHull(points2));

    unordered_set<Vec> points3 = {
        make_vec(2.0,3.0),
        make_vec(2.0,1.0),
    };
    chs.push_back(ConvexHull(points3));

    unordered_set<Vec> points4 = {
        make_vec(1.0,2.0),
    };
    chs.push_back(ConvexHull(points4));

    for (size_t i=0; i<chs.size(); i++) {
        for (size_t j=i+1; j<chs.size(); j++) {
            if (i==0 && j==1) {
                EXPECT_TRUE(chs[i].equals(chs[j]));
                EXPECT_TRUE(chs[j].equals(chs[i]));
                EXPECT_TRUE(chs[i] == chs[j]);
                EXPECT_TRUE(chs[j] == chs[i]);
            } else {
                EXPECT_FALSE(chs[i].equals(chs[j]));
                EXPECT_FALSE(chs[j].equals(chs[i]));
                EXPECT_FALSE(chs[i] == chs[j]);
                EXPECT_FALSE(chs[j] == chs[i]);
            }
        }
    }
}

// /**
//  * Single argument prune is tested in the constructors
//  * Testing the two argument prune funciton
// */
// TEST(Ch_Arithmetic, prune) {
//     TestableConvexHull ch;
//     unordered_set<TaggedPoint> ref_points = {
//         TaggedPoint(make_vec(1.0,3.0), 1),
//         TaggedPoint(make_vec(3.0,1.0), 1),
//     };
//     unordered_set<TaggedPoint> points = {
//         TaggedPoint(make_vec(2.0,3.0), 1),
//         TaggedPoint(make_vec(1.0,1.0), 1),
//         TaggedPoint(make_vec(4.0,0.5), 1),
//         TaggedPoint(make_vec(5.0,0.0), 1),
//         TaggedPoint(make_vec(-1.0,0.0), 1),
//     };
//     unordered_set<TaggedPoint> expected_pruned_points = {
//         TaggedPoint(make_vec(2.0,3.0), 1),
//         TaggedPoint(make_vec(4.0,0.5), 1),
//         TaggedPoint(make_vec(5.0,0.0), 1),
//     };

//     unordered_set<TaggedPoint> pruned_points = ch.public_prune(ref_points, points);
//     EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
// } 

// /**
//  * Single argument prune is tested in the constructors
//  * Testing the two argument prune funciton
// */
// TEST(Ch_Arithmetic, prune_corner_cases) {
//     TestableConvexHull ch;

//     // a point in 'points' is also in 'ref_points' and should be removed
//     unordered_set<TaggedPoint> ref_points = {
//         TaggedPoint(make_vec(1.0,3.0), 1),
//         TaggedPoint(make_vec(3.0,1.0), 1),
//     };
//     unordered_set<TaggedPoint> points = {
//         TaggedPoint(make_vec(2.0,3.0), 1),
//         TaggedPoint(make_vec(1.0,1.0), 1),
//         TaggedPoint(make_vec(1.0,3.0), 1),
//         TaggedPoint(make_vec(4.0,0.5), 1),
//     };
//     unordered_set<TaggedPoint> expected_pruned_points = {
//         TaggedPoint(make_vec(2.0,3.0), 1),
//         TaggedPoint(make_vec(4.0,0.5), 1),
//     };

//     unordered_set<TaggedPoint> pruned_points = ch.public_prune(ref_points, points);
//     EXPECT_TRUE(set_equals(pruned_points, expected_pruned_points));
//     // 'points' contain points that dominate each other, but shouldn't be removed, because not dominated by any 
//     // points in 'ref_points'
//     unordered_set<TaggedPoint> ref_points2 = {
//         TaggedPoint(make_vec(1.0,3.0), 1),
//         TaggedPoint(make_vec(3.0,1.0), 1),
//     };
//     unordered_set<TaggedPoint> points2 = {
//         TaggedPoint(make_vec(0.0,0.0), 1),
//         TaggedPoint(make_vec(1.0,1.0), 1),
//         TaggedPoint(make_vec(2.0,2.0), 1),
//         TaggedPoint(make_vec(3.0,3.0), 1),
//     };
//     unordered_set<TaggedPoint> expected_pruned_points2 = {
//         TaggedPoint(make_vec(2.0,2.0), 1),
//         TaggedPoint(make_vec(3.0,3.0), 1),
//     };

//     unordered_set<TaggedPoint> pruned_points2 = ch.public_prune(ref_points2, points2);
//     EXPECT_TRUE(set_equals(pruned_points2, expected_pruned_points2));
// }

/**
 * 
*/
TEST(Ch_Arithmetic, scale) {
    unordered_set<Vec> points = {
        make_vec(1.0,2.0),
        make_vec(2.0,1.0),
    };
    TestableConvexHull ch(points);

    // test scale
    TestableConvexHull ch1 = ch.scale(0.5);
    unordered_set<Vec> expected_ch1 = {
        make_vec(0.5,1.0),
        make_vec(1.0,0.5),
    };
    EXPECT_TRUE(ch1.check_fits_expected(expected_ch1));
    EXPECT_EQ(ch1.size(), 2u);

    // test operator *
    TestableConvexHull ch2 = (TestableConvexHull) (ch * 2.0);
    unordered_set<Vec> expected_ch2 = {
        make_vec(2.0,4.0),
        make_vec(4.0,2.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected_ch2));
    EXPECT_EQ(ch2.size(), 2u);
}

/**
 * Test one = pf test adapted to have similar result
*/
TEST(Ch_Arithmetic, union_one) {
    unordered_set<Vec> points1 = {
        make_vec(2.0,0.0),
        make_vec(1.1,1.1),
        make_vec(0.0,1.1),
    };
    TestableConvexHull ch1(points1);

    unordered_set<Vec> points2 = {
        make_vec(1.1,1.1),
        make_vec(0.0,2.0),
    };
    TestableConvexHull ch2(points2);

    unordered_set<Vec> expected_union_ch = {
        make_vec(2.0,0.0),
        make_vec(1.1,1.1),
        make_vec(0.0,2.0),
    };

    TestableConvexHull ch3 = (TestableConvexHull) ch1.combine(ch2);
    TestableConvexHull ch4 = (TestableConvexHull) (ch1 | ch2);

    EXPECT_TRUE(ch3.check_fits_expected(expected_union_ch));
    EXPECT_EQ(ch3.size(), 3u);

    EXPECT_TRUE(ch4.check_fits_expected(expected_union_ch));
    EXPECT_EQ(ch4.size(), 3u);
}

/**
 * Test two = pf test, but adapted result
*/
TEST(Ch_Arithmetic, union_two) {
    unordered_set<Vec> points1 = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableConvexHull ch1(points1);

    unordered_set<Vec> points2 = {
        make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };
    TestableConvexHull ch2(points2);

    unordered_set<Vec> expected_union_ch = {
        make_vec(2.0,0.0),
        // make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };

    TestableConvexHull ch3 = (TestableConvexHull) ch1.combine(ch2);
    TestableConvexHull ch4 = (TestableConvexHull) (ch1 | ch2);

    EXPECT_TRUE(ch3.check_fits_expected(expected_union_ch));
    EXPECT_EQ(ch3.size(), 2u);

    EXPECT_TRUE(ch4.check_fits_expected(expected_union_ch));
    EXPECT_EQ(ch4.size(), 2u);
}

/**
 * Test adding
 * There is a dominated vector of (1.0,2.1) from adding 1c and 2a
 * There are two ways of making vector (2.0,2.0) from adding 1b and 2a or adding 1a and 2b
 * 
 * Test one = pf test adapted to have similar result
*/
TEST(Ch_Arithmetic, add_chs_one) {
    unordered_set<Vec> points1 = {
        make_vec(2.1,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableConvexHull ch1(points1);

    unordered_set<Vec> points2 = {
        make_vec(1.0,1.0),
        make_vec(0.0,2.1),
    };
    TestableConvexHull ch2(points2);

    unordered_set<Vec> expected_add_ch = {
        make_vec(3.1,1.0),
        make_vec(2.1,2.1),
        make_vec(1.0,3.1),
        make_vec(0.0,3.2),
    };

    TestableConvexHull ch3 = (TestableConvexHull) ch1.add(ch2);
    TestableConvexHull ch4 = (TestableConvexHull) (ch2 + ch1);

    EXPECT_TRUE(ch3.check_fits_expected(expected_add_ch));
    EXPECT_EQ(ch3.size(), 4u);

    EXPECT_TRUE(ch4.check_fits_expected(expected_add_ch));
    EXPECT_EQ(ch4.size(), 4u);
} 

/**
 * This test is used in pareto fronts, but no longer will have the point (2,2) in the final convex hull
 * 
 * Test two = pf test, but adapted result
*/
TEST(Ch_Arithmetic, add_chs_two) {
    unordered_set<Vec> points1 = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableConvexHull ch1(points1);

    unordered_set<Vec> points2 = { 
        make_vec(1.0,1.0),
        make_vec(0.0,2.0),
    };
    TestableConvexHull ch2(points2);

    unordered_set<Vec> expected_add_ch = {
        make_vec(3.0,1.0),
        // make_vec(2.0,2.0),
        make_vec(1.0,3.0),
        make_vec(0.0,3.1),
    };

    TestableConvexHull ch3 = (TestableConvexHull) ch1.add(ch2);
    TestableConvexHull ch4 = (TestableConvexHull) (ch2 + ch1);

    EXPECT_TRUE(ch3.check_fits_expected(expected_add_ch));
    EXPECT_EQ(ch3.size(), 3u);

    EXPECT_TRUE(ch4.check_fits_expected(expected_add_ch));
    EXPECT_EQ(ch4.size(), 3u);
}

/**
 * 
*/
TEST(Ch_Arithmetic, add_vector) {
    unordered_set<Vec> points = {
        make_vec(2.0,0.0),
        make_vec(1.0,1.0),
        make_vec(0.0,1.1),
    };
    TestableConvexHull ch(points);

    Eigen::ArrayXd v1 = make_vec(1.0,3.0);
    Eigen::ArrayXd v2 = make_vec(-1.0,0.0);

    unordered_set<Vec> expected_ch1 = {
        make_vec(3.0,3.0),
        make_vec(2.0,4.0),
        make_vec(1.0,4.1),
    };

    unordered_set<Vec> expected_ch2 = {
        make_vec(1.0,0.0),
        make_vec(0.0,1.0),
        make_vec(-1.0,1.1),
    };

    TestableConvexHull ch1 = (TestableConvexHull) ch.add(v1);
    TestableConvexHull ch2 = (TestableConvexHull) (ch + v2);

    EXPECT_TRUE(ch1.check_fits_expected(expected_ch1));
    EXPECT_EQ(ch1.size(), 3u);

    EXPECT_TRUE(ch2.check_fits_expected(expected_ch2));
    EXPECT_EQ(ch2.size(), 3u);
}







/**
 * Hypervolume tests
 */
TEST(Ch_Hypervolume, hypervolume_2d) {
    unordered_set<Vec> points = {
        make_vec(2.0,1.0),
        make_vec(1.0,2.0),
    };
    TestableConvexHull ch(points);

    Vec ref_point = make_vec(0.0, 0.0);
    double expected_hypervolume = 3.5;
    EXPECT_NEAR(ch.hypervolume(ref_point), expected_hypervolume, 1.0e-10);
}


// N.B. actually computed this incorrectly by hand originally
// Originally had (1,1,2), thought volume was 6.5, but actually 6.3333, as forgot pyramid at corner
// Changed to (1,1,4) to make hypervolume 11 and not a weird fraction
TEST(Ch_Hypervolume, hypervolume_3d) {
    unordered_set<Vec> points = {
        make_vec(1.0,1.0,4.0),
        make_vec(2.0,2.0,1.0),
    };
    TestableConvexHull ch(points);


    Vec ref_point = make_vec(0.0, 0.0, 0.0);
    double expected_hypervolume = 11.0;
    EXPECT_NEAR(ch.hypervolume(ref_point), expected_hypervolume, 1.0e-10);
}


TEST(Ch_Hypervolume, hypervolume_error) {
    unordered_set<Vec> points = {
        make_vec(1.0,1.0,2.0),
        make_vec(2.0,2.0,1.0),
    };
    TestableConvexHull ch(points);


    Vec ref_point = make_vec(2.0, 2.0, 2.0);
    EXPECT_ANY_THROW(ch.hypervolume(ref_point));
}

/**
 * Test strongly_convex_dominated with a specific case that was causing issues
 * This tests the case where the point to consider is in the ref_points set
 */
TEST(Ch_StronglyConvexDominated, point_in_ref_points) {
    unordered_set<Vec> ref_points = {
        make_vec(16.72, -8),
        make_vec(5.92, -2),
        make_vec(3.96, -4.7),
        make_vec(3.96, -18.5),
        make_vec(2, -32),
        make_vec(3.96, -3),
        make_vec(3.30667, -5.66667),
        make_vec(8.86667, -8.5)
    };
    
    Vec point = make_vec(3.96, -4.7);
    
    // Should ignore the duplicate point, and be dominated by [3.96, -3]
    bool is_dominated = TestableConvexHull::public_strongly_convex_dominated(ref_points, point);
    EXPECT_TRUE(is_dominated);
}