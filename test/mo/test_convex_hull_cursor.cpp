#include "test/mo/test_convex_hull_cursor.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "mo/data_structures/convex_hull.h"

// includes

// #include <qhull>
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/RboxPoints.h"

#include <utility>
#include <unordered_map>
#include <sstream>

#include <Eigen/Dense>

#include "thts_manager.h"

using namespace std;
using namespace thts;
using namespace thts::test;
using namespace orgQhull;


/**
 * Helper function to create 2D vectors
 */
static Eigen::ArrayXd make_vec(double a, double b) {
    Eigen::ArrayXd v(2);
    v[0] = a;
    v[1] = b;
    return v;
}

/**
 * Helper function to create 3D vectors
 */
static Eigen::ArrayXd make_vec(double a, double b, double c) {
    Eigen::ArrayXd v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return v;
}

/**
 * Helper function to create 4D vectors
 */
static Eigen::ArrayXd make_vec(double a, double b, double c, double d) {
    Eigen::ArrayXd v(4);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    return v;
}


// ============================================================================
// CONSTRUCTOR TESTS
// ============================================================================

/**
 * Test empty constructor
 */
TEST(ConvexHull_Constructors, EmptyConstructor) {
    TestableConvexHull ch;
    EXPECT_EQ(ch.size(), 0u);
    EXPECT_EQ(ch.reward_dim(), -1);
}

/**
 * Test constructor from unordered_set<Vec>
 */
TEST(ConvexHull_Constructors, FromUnorderedSetVec) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
        make_vec(1.0, 1.9),  // Should be pruned (dominated by (1.0, 2.0) and (2.0, 1.0))
        make_vec(1.0, 1.0),  // Should be pruned (dominated)
        make_vec(0.0, 0.0),  // Should be pruned (dominated)
    };
    TestableConvexHull ch(points);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
    EXPECT_EQ(ch.reward_dim(), 2);
}

/**
 * Test constructor from unordered_set<Vec> with already_convex_hull=true
 */
TEST(ConvexHull_Constructors, FromUnorderedSetVecAlreadyConvexHull) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points, true);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

// /**
//  * Test constructor from unordered_set<Eigen::ArrayXd>
//  */
// TEST(ConvexHull_Constructors, FromUnorderedSetEigen) {
//     unordered_set<Eigen::ArrayXd> eigen_points = {
//         make_vec(1.0, 2.0),
//         make_vec(2.0, 1.0),
//         make_vec(1.0, 1.5),  // Should be pruned
//     };
//     ConvexHull ch(eigen_points);
//     EXPECT_EQ(ch.size(), 2u);
// }

/**
 * Test constructor from single Vec (heuristic_val)
 */
TEST(ConvexHull_Constructors, FromHeuristicVal) {
    Vec heuristic_val = make_vec(3.0, 4.0);
    TestableConvexHull ch(heuristic_val);
    unordered_set<Vec> expected = {
        make_vec(3.0, 4.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 1u);
    EXPECT_EQ(ch.reward_dim(), 2);
}

/**
 * Test copy constructor
 */
TEST(ConvexHull_Constructors, CopyConstructor) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    TestableConvexHull ch2(ch1);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected));
    EXPECT_EQ(ch2.size(), 2u);
    // Original should be unchanged
    EXPECT_TRUE(ch1.check_fits_expected(expected));
    EXPECT_EQ(ch1.size(), 2u);
}

/**
 * Test move constructor
 */
TEST(ConvexHull_Constructors, MoveConstructor) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    TestableConvexHull ch2(std::move(ch1));
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected));
    EXPECT_EQ(ch2.size(), 2u);
    // Moved-from object should be empty or valid
    EXPECT_EQ(ch1.size(), 0u);
}

/**
 * Test constructor with duplicate points (should be deduplicated)
 */
TEST(ConvexHull_Constructors, DuplicatePoints) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(1.0, 2.0),  // Duplicate
        make_vec(2.0, 1.0),
        make_vec(2.0, 1.0),  // Duplicate
    };
    TestableConvexHull ch(points);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test constructor with single point
 */
TEST(ConvexHull_Constructors, SinglePoint) {
    unordered_set<Vec> points = {
        make_vec(5.0, 3.0),
    };
    TestableConvexHull ch(points);
    unordered_set<Vec> expected = {
        make_vec(5.0, 3.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 1u);
}

/**
 * Test constructor with 3D points
 */
TEST(ConvexHull_Constructors, ThreeDimensionalPoints) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0, 3.0),
        make_vec(2.0, 1.0, 3.0),
        make_vec(1.0, 1.0, 2.0),  // Should be pruned
    };
    TestableConvexHull ch(points);
    EXPECT_GE(ch.size(), 2u);
    EXPECT_EQ(ch.reward_dim(), 3);
}


// ============================================================================
// ASSIGNMENT OPERATOR TESTS
// ============================================================================

/**
 * Test copy assignment operator
 */
TEST(ConvexHull_Assignment, CopyAssignment) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(3.0, 4.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ch2 = ch1;
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected));
    EXPECT_EQ(ch2.size(), 2u);
    // Original should be unchanged
    EXPECT_TRUE(ch1.check_fits_expected(expected));
}

/**
 * Test move assignment operator
 */
TEST(ConvexHull_Assignment, MoveAssignment) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(3.0, 4.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ch2 = std::move(ch1);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch2.check_fits_expected(expected));
    EXPECT_EQ(ch2.size(), 2u);
}

/**
 * Test self-assignment
 */
TEST(ConvexHull_Assignment, SelfAssignment) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    ch = ch;
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}


// ============================================================================
// SCALE OPERATOR TESTS (*=)
// ============================================================================

/**
 * Test *= operator with scalar
 */
TEST(ConvexHull_ScaleOperator, ScaleByScalar) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    ch *= 2.0;
    unordered_set<Vec> expected = {
        make_vec(2.0, 4.0),
        make_vec(4.0, 2.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test *= operator with Vec
 */
TEST(ConvexHull_ScaleOperator, ScaleByVec) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec scale_vec = make_vec(2.0, 3.0);
    ch *= scale_vec;
    unordered_set<Vec> expected = {
        make_vec(2.0, 6.0),
        make_vec(4.0, 3.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test *= operator with zero scalar
 */
TEST(ConvexHull_ScaleOperator, ScaleByZero) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    ch *= 0.0;
    unordered_set<Vec> expected = {
        make_vec(0.0, 0.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 1u);
}

/**
 * Test *= operator with negative scalar
 */
TEST(ConvexHull_ScaleOperator, ScaleByNegative) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    ch *= -1.0;
    unordered_set<Vec> expected = {
        make_vec(-1.0, -2.0),
        make_vec(-2.0, -1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}


// ============================================================================
// UNION OPERATOR TESTS (|=)
// ============================================================================

/**
 * Test |= operator (union/combine)
 */
TEST(ConvexHull_UnionOperator, UnionCombine) {
    unordered_set<Vec> points1 = {
        make_vec(2.0, 0.0),
        make_vec(1.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0, 2.0),
        make_vec(1.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ch1 |= ch2;
    // After union, should contain non-dominated points
    EXPECT_GE(ch1.size(), 2u);
    EXPECT_LE(ch1.size(), 3u);
}

/**
 * Test |= operator with move (union/combine)
 */
TEST(ConvexHull_UnionOperator, UnionCombineMove) {
    unordered_set<Vec> points1 = {
        make_vec(2.0, 0.0),
        make_vec(1.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0, 2.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ch1 |= std::move(ch2);
    EXPECT_GE(ch1.size(), 2u);
}

/**
 * Test |= operator with empty convex hull
 */
TEST(ConvexHull_UnionOperator, UnionWithEmpty) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    TestableConvexHull ch2;  // Empty
    
    ch1 |= ch2;
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch1.check_fits_expected(expected));
    EXPECT_EQ(ch1.size(), 2u);
}


// ============================================================================
// ADDITION OPERATOR TESTS (+=)
// ============================================================================

/**
 * Test += operator with ConvexHull
 */
TEST(ConvexHull_AddOperator, AddConvexHull) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 0.0),
        make_vec(0.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(1.0, 1.0),
        make_vec(0.0, 0.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ch1 += ch2;
    // After addition, should have Minkowski sum points
    EXPECT_GE(ch1.size(), 2u);
}

/**
 * Test += operator with Vec
 */
TEST(ConvexHull_AddOperator, AddVec) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 3.0);
    ch += offset;
    unordered_set<Vec> expected = {
        make_vec(2.0, 5.0),
        make_vec(3.0, 4.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test -= operator with Vec
 */
TEST(ConvexHull_SubtractOperator, SubtractVec) {
    unordered_set<Vec> points = {
        make_vec(2.0, 4.0),
        make_vec(3.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 2.0);
    ch -= offset;
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}


// ============================================================================
// SCALE FUNCTION TESTS
// ============================================================================

/**
 * Test scale(double) function
 */
TEST(ConvexHull_ScaleFunction, ScaleByScalar) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    ConvexHull scaled = ch.scale(0.5);
    unordered_set<Vec> expected = {
        make_vec(0.5, 1.0),
        make_vec(1.0, 0.5),
    };
    TestableConvexHull testable_scaled(scaled);
    EXPECT_TRUE(testable_scaled.check_fits_expected(expected));
    EXPECT_EQ(scaled.size(), 2u);
    // Original should be unchanged
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test scale(Vec) function
 */
TEST(ConvexHull_ScaleFunction, ScaleByVec) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec scale_vec = make_vec(2.0, 3.0);
    ConvexHull scaled = ch.scale(scale_vec);
    unordered_set<Vec> expected = {
        make_vec(2.0, 6.0),
        make_vec(4.0, 3.0),
    };
    TestableConvexHull testable_scaled(scaled);
    EXPECT_TRUE(testable_scaled.check_fits_expected(expected));
    EXPECT_EQ(scaled.size(), 2u);
}


// ============================================================================
// COMBINE FUNCTION TESTS
// ============================================================================

/**
 * Test combine function
 */
TEST(ConvexHull_CombineFunction, CombineTwoHulls) {
    unordered_set<Vec> points1 = {
        make_vec(2.0, 0.0),
        make_vec(1.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0, 2.0),
        make_vec(1.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ConvexHull combined = ch1.combine(ch2);
    EXPECT_GE(combined.size(), 2u);
}

/**
 * Test combine with empty hull
 */
TEST(ConvexHull_CombineFunction, CombineWithEmpty) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    TestableConvexHull ch2;  // Empty
    
    ConvexHull combined = ch1.combine(ch2);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull testable_combined(combined);
    EXPECT_TRUE(testable_combined.check_fits_expected(expected));
    EXPECT_EQ(combined.size(), 2u);
}

/**
 * Test combine commutative property (should call combine on larger set)
 */
TEST(ConvexHull_CombineFunction, CombineCommutative) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 0.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0, 1.0),
        make_vec(0.0, 2.0),
        make_vec(0.0, 3.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ConvexHull combined1 = ch1.combine(ch2);
    ConvexHull combined2 = ch2.combine(ch1);
    
    EXPECT_EQ(combined1.size(), combined2.size());
    EXPECT_TRUE(combined1.equals(combined2));
}


// ============================================================================
// ADD FUNCTION TESTS
// ============================================================================

/**
 * Test add(ConvexHull) function (Minkowski sum)
 */
TEST(ConvexHull_AddFunction, AddTwoHulls) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 0.0),
        make_vec(0.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(1.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ConvexHull sum = ch1.add(ch2);
    // Minkowski sum should have at least 2 points
    EXPECT_GE(sum.size(), 2u);
}

/**
 * Test add(Vec) function
 */
TEST(ConvexHull_AddFunction, AddVec) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 3.0);
    ConvexHull result = ch.add(offset);
    unordered_set<Vec> expected = {
        make_vec(2.0, 5.0),
        make_vec(3.0, 4.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
    EXPECT_EQ(result.size(), 2u);
}

/**
 * Test subtract(Vec) function
 */
TEST(ConvexHull_SubtractFunction, SubtractVec) {
    unordered_set<Vec> points = {
        make_vec(2.0, 4.0),
        make_vec(3.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 2.0);
    ConvexHull result = ch.subtract(offset);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
    EXPECT_EQ(result.size(), 2u);
}


// ============================================================================
// EQUALITY TESTS
// ============================================================================

/**
 * Test equals function
 */
TEST(ConvexHull_Equality, EqualsSamePoints) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    EXPECT_TRUE(ch1.equals(ch2));
    EXPECT_TRUE(ch2.equals(ch1));
    EXPECT_TRUE(ch1 == ch2);
    EXPECT_TRUE(ch2 == ch1);
}

/**
 * Test equals function with different points
 */
TEST(ConvexHull_Equality, EqualsDifferentPoints) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(2.0, 3.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    EXPECT_FALSE(ch1.equals(ch2));
    EXPECT_FALSE(ch2.equals(ch1));
    EXPECT_FALSE(ch1 == ch2);
    EXPECT_FALSE(ch2 == ch1);
}

/**
 * Test equals function with different sizes
 */
TEST(ConvexHull_Equality, EqualsDifferentSizes) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    EXPECT_FALSE(ch1.equals(ch2));
    EXPECT_FALSE(ch2.equals(ch1));
}

/**
 * Test equals function with empty convex hulls
 */
TEST(ConvexHull_Equality, EqualsEmpty) {
    TestableConvexHull ch1;
    TestableConvexHull ch2;
    
    EXPECT_TRUE(ch1.equals(ch2));
    EXPECT_TRUE(ch1 == ch2);
}


// ============================================================================
// SIZE AND DIMENSION TESTS
// ============================================================================

/**
 * Test size function
 */
TEST(ConvexHull_Size, SizeFunction) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
        make_vec(0.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    EXPECT_GE(ch.size(), 2u);
    EXPECT_LE(ch.size(), 3u);
}

/**
 * Test reward_dim function with empty hull
 */
TEST(ConvexHull_RewardDim, EmptyHull) {
    TestableConvexHull ch;
    EXPECT_EQ(ch.reward_dim(), -1);
}

/**
 * Test reward_dim function with 2D points
 */
TEST(ConvexHull_RewardDim, TwoDimensional) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    EXPECT_EQ(ch.reward_dim(), 2);
}

/**
 * Test reward_dim function with 3D points
 */
TEST(ConvexHull_RewardDim, ThreeDimensional) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0, 3.0),
        make_vec(2.0, 1.0, 3.0),
    };
    TestableConvexHull ch(points);
    EXPECT_EQ(ch.reward_dim(), 3);
}


// ============================================================================
// GET BEST POINT TESTS
// ============================================================================

/**
 * Test get_best_point function
 */
TEST(ConvexHull_GetBestPoint, BasicTest) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
        make_vec(0.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(1.0, 0.0);  // Prefer first dimension
    RandManager rand_manager(42);
    
    Vec best = ch.get_best_point(context_weight, rand_manager);
    // With weight (1,0), should prefer point with highest first coordinate
    // That would be (2,1)
    EXPECT_EQ(best.vec[0], 2.0);
    EXPECT_EQ(best.vec[1], 1.0);
}

/**
 * Test get_best_point with different context weights
 */
TEST(ConvexHull_GetBestPoint, DifferentWeights) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(0.0, 1.0);  // Prefer second dimension
    RandManager rand_manager(42);
    
    Vec best = ch.get_best_point(context_weight, rand_manager);
    // With weight (0,1), should prefer point with highest second coordinate
    // That would be (1,2)
    EXPECT_EQ(best.vec[0], 1.0);
    EXPECT_EQ(best.vec[1], 2.0);
}

/**
 * Test get_best_point with ties (should break randomly)
 */
TEST(ConvexHull_GetBestPoint, TiesBreakRandomly) {
    // Points with same dot product: (2,1)·(1,1)=3 and (1,2)·(1,1)=3
    unordered_set<Vec> points = {
        make_vec(2.0, 1.0),
        make_vec(1.0, 2.0),
        make_vec(0.5, 0.5),  // Lower dot product = 1.0
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(1.0, 1.0);  // Equal weight
    RandManager rand_manager(42);
    
    Vec best = ch.get_best_point(context_weight, rand_manager);
    // Should return one of the points with highest dot product (3.0)
    // Either (2,1) or (1,2), both have dot product = 3.0
    EXPECT_TRUE(best == Vec(make_vec(2.0, 1.0)) || best == Vec(make_vec(1.0, 2.0)));
    // Should not be the point with lower dot product
    EXPECT_FALSE(best == Vec(make_vec(0.5, 0.5)));
}


// ============================================================================
// MAX LINEAR UTILITY TESTS
// ============================================================================

/**
 * Test get_max_linear_utility function
 */
TEST(ConvexHull_MaxLinearUtility, BasicTest) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
        make_vec(0.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(1.0, 0.0);
    double max_utility = ch.get_max_linear_utility(context_weight);
    
    // With weight (1,0), max should be max of first coordinates = 2.0
    EXPECT_DOUBLE_EQ(max_utility, 2.0);
}

/**
 * Test get_max_linear_utility with different weights
 */
TEST(ConvexHull_MaxLinearUtility, DifferentWeights) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(0.0, 1.0);
    double max_utility = ch.get_max_linear_utility(context_weight);
    
    // With weight (0,1), max should be max of second coordinates = 2.0
    EXPECT_DOUBLE_EQ(max_utility, 2.0);
}

/**
 * Test get_max_linear_utility with equal weights
 */
TEST(ConvexHull_MaxLinearUtility, EqualWeights) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec context_weight = make_vec(1.0, 1.0);
    double max_utility = ch.get_max_linear_utility(context_weight);
    
    // Dot products: (1,2)·(1,1)=3, (2,1)·(1,1)=3
    EXPECT_DOUBLE_EQ(max_utility, 3.0);
}


// ============================================================================
// HYPERVOLUME TESTS
// ============================================================================

/**
 * Test hypervolume function with 2D points
 */
TEST(ConvexHull_Hypervolume, TwoDimensional) {
    unordered_set<Vec> points = {
        make_vec(2.0, 1.0),
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch(points);
    
    Vec ref_point = make_vec(0.0, 0.0);
    double hv = ch.hypervolume(ref_point);
    
    // Expected hypervolume is 3.5 (as mentioned in code comments)
    EXPECT_NEAR(hv, 3.5, 1e-9);
}

/**
 * Test hypervolume function with 3D points
 */
TEST(ConvexHull_Hypervolume, ThreeDimensional) {
    unordered_set<Vec> points = {
        make_vec(1.0, 1.0, 4.0),
        make_vec(2.0, 2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec ref_point = make_vec(0.0, 0.0, 0.0);
    double hv = ch.hypervolume(ref_point);
    
    // Expected hypervolume is 11.0 (as mentioned in code comments)
    EXPECT_NEAR(hv, 11.0, 1e-9);
}

/**
 * Test hypervolume function with single point
 */
TEST(ConvexHull_Hypervolume, SinglePoint) {
    unordered_set<Vec> points = {
        make_vec(2.0, 2.0),
    };
    TestableConvexHull ch(points);
    
    Vec ref_point = make_vec(0.0, 0.0);
    double hv = ch.hypervolume(ref_point);
    
    // Hypervolume should be 4.0 (2*2 rectangle)
    EXPECT_NEAR(hv, 4.0, 1e-9);
}

/**
 * Test hypervolume function with empty hull
 */
TEST(ConvexHull_Hypervolume, EmptyHull) {
    TestableConvexHull ch;
    
    Vec ref_point = make_vec(0.0, 0.0);
    double hv = ch.hypervolume(ref_point);
    
    EXPECT_DOUBLE_EQ(hv, 0.0);
}

/**
 * Test hypervolume function throws error when ref_point not dominated
 */
TEST(ConvexHull_Hypervolume, ErrorWhenRefPointNotDominated) {
    unordered_set<Vec> points = {
        make_vec(1.0, 1.0, 2.0),
        make_vec(2.0, 2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec ref_point = make_vec(2.0, 2.0, 2.0);  // Not dominated by points
    EXPECT_THROW(ch.hypervolume(ref_point), runtime_error);
}


// ============================================================================
// SPARSITY METRIC TESTS
// ============================================================================

/**
 * Test sparsity_metric function with multiple points
 */
TEST(ConvexHull_SparsityMetric, MultiplePoints) {
    unordered_set<Vec> points = {
        make_vec(1.0, 0.0),
        make_vec(2.0, 0.0),
        make_vec(3.0, 0.0),
    };
    TestableConvexHull ch(points);
    
    double sparsity = ch.sparsity_metric();
    
    // Sparsity should be non-negative
    EXPECT_GE(sparsity, 0.0);
}

/**
 * Test sparsity_metric function with single point
 */
TEST(ConvexHull_SparsityMetric, SinglePoint) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch(points);
    
    double sparsity = ch.sparsity_metric();
    
    // With < 2 points, sparsity should be 0
    EXPECT_DOUBLE_EQ(sparsity, 0.0);
}

/**
 * Test sparsity_metric function with empty hull
 */
TEST(ConvexHull_SparsityMetric, EmptyHull) {
    TestableConvexHull ch;
    
    double sparsity = ch.sparsity_metric();
    
    // With < 2 points, sparsity should be 0
    EXPECT_DOUBLE_EQ(sparsity, 0.0);
}


// ============================================================================
// ADDITIVE EPSILON METRIC TESTS
// ============================================================================

/**
 * Test additive_eps_metric function with multiple points
 */
TEST(ConvexHull_AdditiveEpsMetric, MultiplePoints) {
    unordered_set<Vec> points = {
        make_vec(1.0, 0.0),
        make_vec(0.0, 1.0),
        make_vec(0.5, 0.5),
    };
    TestableConvexHull ch(points);
    
    double eps = ch.additive_eps_metric();
    
    // Epsilon should be non-negative
    EXPECT_GE(eps, 0.0);
}

/**
 * Test additive_eps_metric function with single point
 */
TEST(ConvexHull_AdditiveEpsMetric, SinglePoint) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch(points);
    
    double eps = ch.additive_eps_metric();
    
    // With < 2 points, epsilon should be 0
    EXPECT_DOUBLE_EQ(eps, 0.0);
}

/**
 * Test additive_eps_metric function with empty hull
 */
TEST(ConvexHull_AdditiveEpsMetric, EmptyHull) {
    TestableConvexHull ch;
    
    double eps = ch.additive_eps_metric();
    
    // With < 2 points, epsilon should be 0
    EXPECT_DOUBLE_EQ(eps, 0.0);
}


// ============================================================================
// PRUNE FUNCTION TESTS
// ============================================================================

/**
 * Test prune function
 */
TEST(ConvexHull_Prune, BasicPrune) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
        make_vec(1.0, 1.9),  // Should be pruned
        make_vec(1.0, 1.0),  // Should be pruned
        make_vec(0.0, 0.0),  // Should be pruned
    };
    TestableConvexHull ch;
    
    unordered_set<Vec> pruned = ch.public_prune(points);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    
    EXPECT_TRUE(set_equals(pruned, expected));
}

/**
 * Test prune function with already pruned points
 */
TEST(ConvexHull_Prune, AlreadyPruned) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch;
    
    unordered_set<Vec> pruned = ch.public_prune(points);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    
    EXPECT_TRUE(set_equals(pruned, expected));
}

/**
 * Test prune function with single point
 */
TEST(ConvexHull_Prune, SinglePoint) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch;
    
    unordered_set<Vec> pruned = ch.public_prune(points);
    unordered_set<Vec> expected = {
        make_vec(1.0, 2.0),
    };
    
    EXPECT_TRUE(set_equals(pruned, expected));
}

/**
 * Test prune function with empty set
 */
TEST(ConvexHull_Prune, EmptySet) {
    unordered_set<Vec> points;
    TestableConvexHull ch;
    
    unordered_set<Vec> pruned = ch.public_prune(points);
    
    EXPECT_EQ(pruned.size(), 0u);
}


// ============================================================================
// STRONGLY CONVEX DOMINATED TESTS
// ============================================================================

/**
 * Test strongly_convex_dominated function
 */
TEST(ConvexHull_StronglyConvexDominated, BasicTest) {
    unordered_set<Vec> ref_points = {
        make_vec(2.0, 0.0),
        make_vec(0.0, 2.0),
    };
    
    Vec point1 = make_vec(1.0, 1.0);  // Should be dominated
    Vec point2 = make_vec(3.0, 3.0);  // Should not be dominated
    
    bool dominated1 = TestableConvexHull::public_strongly_convex_dominated(ref_points, point1);
    bool dominated2 = TestableConvexHull::public_strongly_convex_dominated(ref_points, point2);
    
    EXPECT_TRUE(dominated1);
    EXPECT_FALSE(dominated2);
}

/**
 * Test strongly_convex_dominated with point in ref_points
 */
TEST(ConvexHull_StronglyConvexDominated, PointInRefPoints) {
    unordered_set<Vec> ref_points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    
    Vec point = make_vec(1.0, 2.0);  // Same as one ref point
    
    bool dominated = TestableConvexHull::public_strongly_convex_dominated(ref_points, point);
    
    // Should not be dominated by itself
    EXPECT_FALSE(dominated);
}

/**
 * Test strongly_convex_dominated with empty ref_points
 */
TEST(ConvexHull_StronglyConvexDominated, EmptyRefPoints) {
    unordered_set<Vec> ref_points;
    
    Vec point = make_vec(1.0, 2.0);
    
    bool dominated = TestableConvexHull::public_strongly_convex_dominated(ref_points, point);
    
    // Cannot be dominated by nothing
    EXPECT_FALSE(dominated);
}

/**
 * Test strongly_convex_dominated with single ref_point
 */
TEST(ConvexHull_StronglyConvexDominated, SingleRefPoint) {
    unordered_set<Vec> ref_points = {
        make_vec(2.0, 2.0),
    };
    
    Vec point1 = make_vec(1.0, 1.0);  // Should be dominated
    Vec point2 = make_vec(3.0, 3.0);  // Should not be dominated
    
    bool dominated1 = TestableConvexHull::public_strongly_convex_dominated(ref_points, point1);
    bool dominated2 = TestableConvexHull::public_strongly_convex_dominated(ref_points, point2);
    
    EXPECT_TRUE(dominated1);
    EXPECT_FALSE(dominated2);
}


// ============================================================================
// OPERATOR OVERLOADS TESTS (non-member functions)
// ============================================================================

/**
 * Test operator* with scalar on left
 */
TEST(ConvexHull_OperatorOverloads, ScalarTimesHull) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    
    ConvexHull result = 2.0 * ch1;
    unordered_set<Vec> expected = {
        make_vec(2.0, 4.0),
        make_vec(4.0, 2.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
}

/**
 * Test operator* with Vec on left
 */
TEST(ConvexHull_OperatorOverloads, VecTimesHull) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch1(points);
    
    Vec scale_vec = make_vec(2.0, 3.0);
    ConvexHull result = scale_vec * ch1;
    unordered_set<Vec> expected = {
        make_vec(2.0, 6.0),
        make_vec(4.0, 3.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
}

/**
 * Test operator| (union)
 */
TEST(ConvexHull_OperatorOverloads, UnionOperator) {
    unordered_set<Vec> points1 = {
        make_vec(2.0, 0.0),
        make_vec(1.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(0.0, 2.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ConvexHull result = ch1 | ch2;
    EXPECT_GE(result.size(), 2u);
}

/**
 * Test operator+ with two convex hulls
 */
TEST(ConvexHull_OperatorOverloads, AddHulls) {
    unordered_set<Vec> points1 = {
        make_vec(1.0, 0.0),
        make_vec(0.0, 1.0),
    };
    unordered_set<Vec> points2 = {
        make_vec(1.0, 1.0),
    };
    TestableConvexHull ch1(points1);
    TestableConvexHull ch2(points2);
    
    ConvexHull result = ch1 + ch2;
    EXPECT_GE(result.size(), 2u);
}

/**
 * Test operator+ with Vec on left
 */
TEST(ConvexHull_OperatorOverloads, VecPlusHull) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 3.0);
    ConvexHull result = offset + ch;
    unordered_set<Vec> expected = {
        make_vec(2.0, 5.0),
        make_vec(3.0, 4.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
}

/**
 * Test operator- with Vec on left
 */
TEST(ConvexHull_OperatorOverloads, VecMinusHull) {
    unordered_set<Vec> points = {
        make_vec(2.0, 4.0),
        make_vec(3.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    Vec offset = make_vec(1.0, 2.0);
    ConvexHull result = offset - ch;
    unordered_set<Vec> expected = {
        make_vec(-1.0, -2.0),
        make_vec(-2.0, -1.0),
    };
    TestableConvexHull testable_result(result);
    EXPECT_TRUE(testable_result.check_fits_expected(expected));
}


// ============================================================================
// OUTPUT STREAM TESTS
// ============================================================================

/**
 * Test write_to_ostream function
 */
TEST(ConvexHull_Output, WriteToOStream) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
        make_vec(2.0, 1.0),
    };
    TestableConvexHull ch(points);
    
    ostringstream oss;
    ch.write_to_ostream(oss);
    string output = oss.str();
    
    // Should contain "ConvexHull = {" and "}"
    EXPECT_TRUE(output.find("ConvexHull") != string::npos);
    EXPECT_TRUE(output.find("{") != string::npos);
    EXPECT_TRUE(output.find("}") != string::npos);
}

/**
 * Test operator<< for output stream
 */
TEST(ConvexHull_Output, StreamOperator) {
    unordered_set<Vec> points = {
        make_vec(1.0, 2.0),
    };
    TestableConvexHull ch(points);
    
    ostringstream oss;
    oss << ch;
    string output = oss.str();
    
    // Should contain "ConvexHull"
    EXPECT_TRUE(output.find("ConvexHull") != string::npos);
}


// ============================================================================
// EDGE CASE TESTS
// ============================================================================

/**
 * Test with very small values
 */
TEST(ConvexHull_EdgeCases, VerySmallValues) {
    unordered_set<Vec> points = {
        make_vec(1e-10, 2e-10),
        make_vec(2e-10, 1e-10),
    };
    TestableConvexHull ch(points);
    
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test with very large values
 */
TEST(ConvexHull_EdgeCases, VeryLargeValues) {
    unordered_set<Vec> points = {
        make_vec(1e10, 2e10),
        make_vec(2e10, 1e10),
    };
    TestableConvexHull ch(points);
    
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test with negative values
 */
TEST(ConvexHull_EdgeCases, NegativeValues) {
    unordered_set<Vec> points = {
        make_vec(-1.0, -2.0),
        make_vec(-2.0, -1.0),
        make_vec(-1.5, -1.5),  // Should be pruned
    };
    TestableConvexHull ch(points);
    
    unordered_set<Vec> expected = {
        make_vec(-1.0, -2.0),
        make_vec(-2.0, -1.0),
    };
    EXPECT_TRUE(ch.check_fits_expected(expected));
    EXPECT_EQ(ch.size(), 2u);
}

/**
 * Test with colinear points
 */
TEST(ConvexHull_EdgeCases, ColinearPoints) {
    unordered_set<Vec> points = {
        make_vec(0.0, 0.0),
        make_vec(1.0, 1.0),
        make_vec(2.0, 2.0),
        make_vec(3.0, 3.0),
    };
    TestableConvexHull ch(points);
    
    // Should prune to just the extreme points
    EXPECT_LE(ch.size(), 2u);
}
