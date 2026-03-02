#include "test/mo/test_simplex_map.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "mo/data_structures/simplex_map.h"
#include "mo/mo_helper.h"
#include "thts_manager.h"

#include <cmath>
#include <memory>
#include <vector>

#include <iostream>

using namespace std;
using namespace thts;
using namespace thts::test;

// ---------------------------------------------------------------------------
// AUTO GENERATED TESTS
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Vec make_vec(double a, double b) {
    Eigen::ArrayXd v(2);
    v[0] = a;
    v[1] = b;
    return Vec(std::move(v));
}

static Vec make_vec(double a, double b, double c) {
    Eigen::ArrayXd v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return Vec(std::move(v));
}

// ---------------------------------------------------------------------------
// SMRegistry tests
// ---------------------------------------------------------------------------

TEST(SimplexMap_SMRegistry, get_or_create_vertex_returns_same_for_same_weight) {
    SMRegistry registry;
    Vec weight = make_vec(1.0, 0.0);
    Vec value = make_vec(0.5, 0.5);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(weight, value, 0.0);
    shared_ptr<SMVertex> v2 = registry.get_or_create_vertex(weight, value, 0.0);
    EXPECT_EQ(v1, v2);
    EXPECT_TRUE(v1->weight.equals(weight));
}

TEST(SimplexMap_SMRegistry, get_or_create_vertex_different_weights_different_vertices) {
    SMRegistry registry;
    Vec value = make_vec(0.5, 0.5);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v2 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    EXPECT_NE(v1, v2);
}

TEST(SimplexMap_SMRegistry, get_or_create_vertex_midpoint_returns_same_for_same_pair_and_ratio) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMVertex> mid1 = registry.get_or_create_vertex(v0, v1, 0.5);
    shared_ptr<SMVertex> mid2 = registry.get_or_create_vertex(v0, v1, 0.5);
    EXPECT_EQ(mid1, mid2);
}

TEST(SimplexMap_SMRegistry, get_or_create_edge_returns_same_for_same_pair) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> e1 = registry.get_or_create_edge(v0, v1);
    shared_ptr<SMEdge> e2 = registry.get_or_create_edge(v0, v1);
    EXPECT_EQ(e1, e2);
}

TEST(SimplexMap_SMRegistry, get_or_create_edge_symmetric_pair_returns_same_edge) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> e1 = registry.get_or_create_edge(v0, v1);
    shared_ptr<SMEdge> e2 = registry.get_or_create_edge(v1, v0);
    EXPECT_EQ(e1, e2);
}

// ---------------------------------------------------------------------------
// SMVertex tests (via registry)
// ---------------------------------------------------------------------------

TEST(SimplexMap_SMVertex, hash_and_equals_consistent) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v2 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    EXPECT_EQ(v1->hash(), v2->hash());
    EXPECT_TRUE(v1->equals(*v2));
}

TEST(SimplexMap_SMVertex, add_bidirectional_connection_adds_to_both_neighbours) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    v0->add_bidirectional_connection(v1);
    EXPECT_TRUE(v0->neighbours->count(v1) > 0);
    EXPECT_TRUE(v1->neighbours->count(v0) > 0);
}

TEST(SimplexMap_SMVertex, erase_bidirectional_connection_removes_from_both) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    v0->add_bidirectional_connection(v1);
    v0->erase_bidirectional_connection(v1);
    EXPECT_EQ(v0->neighbours->count(v1), 0u);
    EXPECT_EQ(v1->neighbours->count(v0), 0u);
}

// ---------------------------------------------------------------------------
// SMEdge tests
// ---------------------------------------------------------------------------

TEST(SimplexMap_SMEdge, hash_and_equals_consistent) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> e1 = registry.get_or_create_edge(v0, v1);
    shared_ptr<SMEdge> e2 = registry.get_or_create_edge(v0, v1);
    EXPECT_EQ(e1->hash(), e2->hash());
    EXPECT_TRUE(e1->equals(*e2));
}

TEST(SimplexMap_SMEdge, find_closest_point_on_edge_ratio_endpoints) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> edge = registry.get_or_create_edge(v0, v1);
    double ratio0 = edge->find_closest_point_on_edge_ratio(v0->weight);
    double ratio1 = edge->find_closest_point_on_edge_ratio(v1->weight);
    EXPECT_NEAR(ratio0, 0.0, 1e-10);
    EXPECT_NEAR(ratio1, 1.0, 1e-10);
}

TEST(SimplexMap_SMEdge, find_closest_point_on_edge_midpoint) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> edge = registry.get_or_create_edge(v0, v1);
    Vec mid = make_vec(0.5, 0.5);
    Vec closest = edge->find_closest_point_on_edge(mid);
    EXPECT_NEAR(closest.vec[0], 0.5, 1e-10);
    EXPECT_NEAR(closest.vec[1], 0.5, 1e-10);
}

TEST(SimplexMap_SMEdge, get_edge_partition_leaf_returns_self) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> edge = registry.get_or_create_edge(v0, v1);
    shared_ptr<unordered_set<shared_ptr<SMEdge>>> partition = edge->get_edge_partition();
    EXPECT_EQ(partition->size(), 1u);
    EXPECT_TRUE(partition->count(edge) > 0);
}

TEST(SimplexMap_SMEdge, split_creates_midpoint_and_two_children) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    shared_ptr<SMEdge> edge = registry.get_or_create_edge(v0, v1);
    edge->split(registry);
    EXPECT_NE(edge->midpoint, nullptr);
    EXPECT_NE(edge->child_edge_0, nullptr);
    EXPECT_NE(edge->child_edge_1, nullptr);
    EXPECT_TRUE(edge->midpoint->weight.equals(make_vec(0.5, 0.5)));
}

// ---------------------------------------------------------------------------
// SMSimplex tests
// ---------------------------------------------------------------------------

TEST(SimplexMap_SMSimplex, constructor_2d_creates_leaf_simplex) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    vector<shared_ptr<SMVertex>> vertices = {
        registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0),
        registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0)
    };
    SMSimplex simplex(2, vertices, 0);
    EXPECT_EQ(simplex.dim, 2);
    EXPECT_EQ(simplex.vertices.size(), 2u);
    EXPECT_TRUE(simplex.is_leaf());
    EXPECT_TRUE(simplex.is_2d());
}

TEST(SimplexMap_SMSimplex, get_closest_vertex_returns_vertex_for_corner_weight) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    vector<shared_ptr<SMVertex>> vertices = {
        registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0),
        registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0)
    };
    SMSimplex simplex(2, vertices, 0);
    shared_ptr<SMVertex> closest = simplex.get_closest_vertex(make_vec(1.0, 0.0));
    EXPECT_TRUE(closest->weight.equals(make_vec(1.0, 0.0)));
}

TEST(SimplexMap_SMSimplex, contains_vertex) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    shared_ptr<SMVertex> v0 = registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0);
    shared_ptr<SMVertex> v1 = registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0);
    vector<shared_ptr<SMVertex>> vertices = { v0, v1 };
    SMSimplex simplex(2, vertices, 0);
    EXPECT_TRUE(simplex.contains_vertex(v0));
    EXPECT_TRUE(simplex.contains_vertex(v1));
    shared_ptr<SMVertex> v_other = registry.get_or_create_vertex(make_vec(0.5, 0.5), value, 0.0);
    EXPECT_FALSE(simplex.contains_vertex(v_other));
}

TEST(SimplexMap_SMSimplex, create_children_subdivides_simplex) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    vector<shared_ptr<SMVertex>> vertices = {
        registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0),
        registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0)
    };
    SMSimplex simplex(2, vertices, 0);
    simplex.create_children(registry);
    EXPECT_FALSE(simplex.is_leaf());
    EXPECT_NE(simplex.normal_child, nullptr);
    EXPECT_NE(simplex.opposite_child, nullptr);
    EXPECT_NE(simplex.split_vertex, nullptr);
}

TEST(SimplexMap_SMSimplex, traverse_returns_correct_child) {
    SMRegistry registry;
    Vec value = make_vec(0.0, 0.0);
    vector<shared_ptr<SMVertex>> vertices = {
        registry.get_or_create_vertex(make_vec(1.0, 0.0), value, 0.0),
        registry.get_or_create_vertex(make_vec(0.0, 1.0), value, 0.0)
    };
    SMSimplex simplex(2, vertices, 0);
    simplex.create_children(registry);
    Vec weight_near_v0 = make_vec(0.9, 0.1);
    shared_ptr<SMSimplex> child = simplex.traverse(weight_near_v0);
    EXPECT_NE(child, nullptr);
}

// ---------------------------------------------------------------------------
// SMMesh / SimplexMap tests
// ---------------------------------------------------------------------------

TEST(SimplexMap_SMMesh, constructor_and_initialise_mesh_2d) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    EXPECT_NE(mesh.root_simplex, nullptr);
    EXPECT_EQ(mesh.all_vertices_vector.size(), 2u);
    EXPECT_TRUE(mesh.root_simplex->is_leaf());
}

TEST(SimplexMap_SMMesh, get_simplex_returns_root_for_2d_before_subdivision) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    Vec weight = make_vec(0.3, 0.7);
    shared_ptr<SMSimplex> simplex = mesh.get_simplex(weight);
    EXPECT_EQ(simplex, mesh.root_simplex);
}

TEST(SimplexMap_SMMesh, get_closest_vertex_2d) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    shared_ptr<SMVertex> closest = mesh.get_closest_vertex(make_vec(1.0, 0.0));
    EXPECT_TRUE(closest->weight.equals(make_vec(1.0, 0.0)));
}

TEST(SimplexMap_SMMesh, get_closest_vertex_3d) {
    SMMesh mesh(3, true, true, true);
    Vec heuristic = Vec(3);
    mesh.initialise_mesh(heuristic);
    shared_ptr<SMVertex> closest = mesh.get_closest_vertex(make_vec(0.75, 0.25, 0.0));
    EXPECT_TRUE(closest->weight.equals(make_vec(1.0, 0.0, 0.0)));
}

TEST(SimplexMap_SMMesh, sample_random_vertex_returns_valid_vertex) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    RandManager rand_manager(42);
    for (int i = 0; i < 20; ++i) {
        shared_ptr<SMVertex> v = mesh.sample_random_vertex(rand_manager);
        EXPECT_NE(v, nullptr);
        EXPECT_EQ(v->weight.size(), 2);
    }
}

TEST(SimplexMap_SMMesh, get_value_estimate_initial) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(1.0, 2.0);
    mesh.initialise_mesh(heuristic);
    shared_ptr<SMVertex> v = mesh.get_closest_vertex(make_vec(1.0, 0.0));
    Vec value = mesh.get_value_estimate(v);
    EXPECT_EQ(value.size(), 2);
    EXPECT_NEAR(value.vec[0], 0.0, 1e-10);
    EXPECT_NEAR(value.vec[1], 0.0, 1e-10);
    Vec value_for_search = mesh.get_value_estimate_for_search(v);
    EXPECT_EQ(value_for_search.size(), 2);
    EXPECT_NEAR(value_for_search.vec[0], 1.0, 1e-10);
    EXPECT_NEAR(value_for_search.vec[1], 2.0, 1e-10);
}

TEST(SimplexMap_SMMesh, update_vertex_values_and_share) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    RandManager rand_manager(42);
    shared_ptr<SMVertex> v = mesh.get_closest_vertex(make_vec(1.0, 0.0));
    Vec new_value = make_vec(5.0, 10.0);
    mesh.update_vertex_values_and_share(rand_manager, v, 1, -1, new_value, new_value, 0.0);
    Vec value = mesh.get_value_estimate(v);
    EXPECT_NEAR(value.vec[0], 5.0, 1e-10);
    EXPECT_NEAR(value.vec[1], 10.0, 1e-10);
}

TEST(SimplexMap_SMMesh, get_pretty_print_string_non_empty) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    string s = mesh.get_pretty_print_string();
    EXPECT_FALSE(s.empty());
}

TEST(SimplexMap_SMMesh, get_approximate_convex_hull_2d_after_update) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    RandManager rand_manager(42);
    shared_ptr<SMVertex> v = mesh.get_closest_vertex(make_vec(1.0, 0.0));
    mesh.update_vertex_values_and_share(rand_manager, v, 1, -1, make_vec(1.0, 0.0), make_vec(1.0, 0.0), 0.0);
    ConvexHull ch = mesh.get_approximate_convex_hull();
    EXPECT_GE(ch.size(), 1u);
}

TEST(SimplexMap_SMMesh, maybe_subdivide_does_not_subdivide_uniform_values) {
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(1.0, 1.0);
    mesh.initialise_mesh(heuristic);
    shared_ptr<SMSimplex> root = mesh.root_simplex;
    RandManager rand_manager(42);
    shared_ptr<SMVertex> v = mesh.get_closest_vertex(make_vec(0.5, 0.5));
    mesh.update_vertex_values_and_share(rand_manager, v, 1, -1, heuristic, heuristic, 0.0);
    mesh.maybe_subdivide(root, 0.01, 10, 2);
    // With uniform values, root may still be leaf (subdivision depends on value variance)
    EXPECT_NE(mesh.root_simplex, nullptr);
}

TEST(SimplexMap_SMMesh, initialise_mesh_3d_creates_unit_simplex) {
    SMMesh mesh(3, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    EXPECT_NE(mesh.root_simplex, nullptr);
    EXPECT_EQ(mesh.all_vertices_vector.size(), 3u);
    EXPECT_TRUE(mesh.root_simplex->is_leaf());
}






// ---------------------------------------------------------------------------
// HUMAN WRITTEN TESTS
// ---------------------------------------------------------------------------

// Auto generated tests are good for checking some basic functionality, but hasn't captured complexity of class
// Specifically a lot of functionality only exists for 3+ dimensions

// helper - checks SMVertex neighbourhood graph is as expected
void sanity_check_2d_smvertex_graph(SMMesh& mesh)
{
    for (shared_ptr<SMVertex> vertex : mesh.all_vertices_vector)
    {
        if (vertex->weight.equals(make_vec(1.0, 0.0)) || vertex->weight.equals(make_vec(0.0, 1.0)))
        {
            EXPECT_EQ(vertex->neighbours->size(), 1u);
        }
        else
        {
            EXPECT_EQ(vertex->neighbours->size(), 2u);
        }
    }
}


// one more comprehensive test for 2D, that all the behaviour leads to a simplex map as expected
TEST(SimplexMap_SMMesh2D, subdivision_works_as_expected)
{
    // [0,1] -> [1,0]
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);

    // Split
    shared_ptr<SMSimplex> root = mesh.root_simplex;
    mesh.subdivide_simplex_test(root, 0.01, 10, 2);

    // Check current state (thoroughly for first split)
    // [0,1] -> [0.5,0.5] -> [1,0]
    EXPECT_NE(root->normal_child, nullptr);
    EXPECT_NE(root->opposite_child, nullptr);
    EXPECT_NE(root->split_vertex, nullptr);
    EXPECT_TRUE(root->normal_child->is_leaf());
    EXPECT_TRUE(root->opposite_child->is_leaf());
    EXPECT_TRUE(root->split_vertex->weight.equals(make_vec(0.5, 0.5)));
    EXPECT_EQ(root->opposite_child->longest_edge.first->weight, make_vec(0.0, 1.0));
    EXPECT_EQ(root->opposite_child->longest_edge.second->weight, make_vec(0.5, 0.5));
    EXPECT_EQ(root->normal_child->longest_edge.first->weight, make_vec(0.5, 0.5));
    EXPECT_EQ(root->normal_child->longest_edge.second->weight, make_vec(1.0, 0.0));

    // And coarsely
    EXPECT_EQ(mesh.all_vertices_vector.size(), 3u);
    EXPECT_EQ(mesh.all_vertices_set.size(), 3u);
    sanity_check_2d_smvertex_graph(mesh);

    // Lookup [0.7,0.3] to get [0.5,0.5]
    Vec context_weight = make_vec(0.7, 0.3);
    shared_ptr<SMVertex> vertex = mesh.get_closest_vertex(context_weight);
    EXPECT_TRUE(vertex->weight.equals(make_vec(0.5, 0.5)));

    // Split again
    shared_ptr<SMSimplex> r_child = root->normal_child;
    mesh.subdivide_simplex_test(r_child, 0.01, 10, 2);

    // Check current state
    // [0,1] -> [0.5,0.5] -> [0.75,0.25] -> [1,0]
    EXPECT_EQ(mesh.all_vertices_vector.size(), 4u);
    EXPECT_EQ(mesh.all_vertices_set.size(), 4u);
    sanity_check_2d_smvertex_graph(mesh);

    // Lookup [0.7,0.3] to get [0.75,0.25]
    vertex = mesh.get_closest_vertex(context_weight);
    EXPECT_TRUE(vertex->weight.equals(make_vec(0.75, 0.25)));

    // Split again a few more times
    shared_ptr<SMSimplex> l_child = root->opposite_child;
    mesh.subdivide_simplex_test(l_child, 0.01, 10, 2);
    shared_ptr<SMSimplex> rl_child = r_child->opposite_child;
    mesh.subdivide_simplex_test(rl_child, 0.01, 10, 2);
    shared_ptr<SMSimplex> rr_child = r_child->normal_child;
    mesh.subdivide_simplex_test(rr_child, 0.01, 10, 2);

    // Check current state
    // [0,1] -> [0.25,0.75] -> [0.5,0.5] -> [0.625,0.375] -> [0.75,0.25] -> [0.875,0.125] -> [1,0]
    EXPECT_EQ(mesh.all_vertices_vector.size(), 7u);
    EXPECT_EQ(mesh.all_vertices_set.size(), 7u);
    sanity_check_2d_smvertex_graph(mesh);

    // Lookup [0.7,0.3] to still get [0.75,0.25]
    vertex = mesh.get_closest_vertex(context_weight);
    EXPECT_TRUE(vertex->weight.equals(make_vec(0.75, 0.25)));
}

// also check that message passing works in 2D as expected
TEST(SimplexMap_SMMesh2D, test_message_passing)
{
    // Copy setup (without checks) from previous test
    // To get mesh: [0,1] -> [0.25,0.75] -> [0.5,0.5] -> [0.625,0.375] -> [0.75,0.25] -> [0.875,0.125] -> [1,0]
    SMMesh mesh(2, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0);
    mesh.initialise_mesh(heuristic);
    
    shared_ptr<SMSimplex> root = mesh.root_simplex;
    mesh.subdivide_simplex_test(root, 0.01, 10, 2);
    shared_ptr<SMSimplex> r_child = root->normal_child;
    mesh.subdivide_simplex_test(r_child, 0.01, 10, 2);
    shared_ptr<SMSimplex> l_child = root->opposite_child;
    mesh.subdivide_simplex_test(l_child, 0.01, 10, 2);
    shared_ptr<SMSimplex> rl_child = r_child->opposite_child;
    mesh.subdivide_simplex_test(rl_child, 0.01, 10, 2);
    shared_ptr<SMSimplex> rr_child = r_child->normal_child;
    mesh.subdivide_simplex_test(rr_child, 0.01, 10, 2);

    // Lookup [0.2,0.8] to get vertex at [0.25,0.75]
    Vec context_weight = make_vec(0.2, 0.8);
    shared_ptr<SMVertex> vertex = mesh.get_closest_vertex(context_weight);
    
    // Update value estimate to something unique and checkable
    RandManager rand_manager(42);
    Vec new_value = make_vec(10.0, 20.0);
    Vec new_value_for_search = make_vec(20.0, 10.0);
    double new_entropy_estimate = 1.0;
    mesh.update_vertex_values_and_share(
        rand_manager, 
        vertex, 
        2, //max push radius = 2
        -1, new_value, new_value_for_search, new_entropy_estimate);

    // Vertices we expect to be updated:
    Vec val_to_ignore = make_vec(-1.0, -1.0);
     double entropy_to_ignore = -1.0;

    Vec u0 = make_vec(0.0, 1.0);
    Vec u1 = make_vec(0.25, 0.75);
    Vec u2 = make_vec(0.5, 0.5);
    Vec u3 = make_vec(0.625, 0.375);

    Vec z0 = make_vec(0.75, 0.25);
    Vec z1 = make_vec(0.875, 0.125);
    Vec z2 = make_vec(1.0, 0.0);

    shared_ptr<SMVertex> v_u0 = mesh.registry.get_or_create_vertex(u0, val_to_ignore, entropy_to_ignore);
    shared_ptr<SMVertex> v_u1 = mesh.registry.get_or_create_vertex(u1, val_to_ignore, entropy_to_ignore);
    shared_ptr<SMVertex> v_u2 = mesh.registry.get_or_create_vertex(u2, val_to_ignore, entropy_to_ignore);
    shared_ptr<SMVertex> v_u3 = mesh.registry.get_or_create_vertex(u3, val_to_ignore, entropy_to_ignore);

    shared_ptr<SMVertex> v_z0 = mesh.registry.get_or_create_vertex(z0, val_to_ignore, entropy_to_ignore);
    shared_ptr<SMVertex> v_z1 = mesh.registry.get_or_create_vertex(z1, val_to_ignore, entropy_to_ignore);
    shared_ptr<SMVertex> v_z2 = mesh.registry.get_or_create_vertex(z2, val_to_ignore, entropy_to_ignore);

    vector<shared_ptr<SMVertex>> expected_updated_vertices = 
    {
        v_u0,
        v_u1,
        v_u2,
        v_u3,
    };
    vector<shared_ptr<SMVertex>> expected_unchanged_vertices = 
    {
        v_z0,
        v_z1,
        v_z2,
    };

    // Check value estimate is updated
    for (shared_ptr<SMVertex> vertex_to_check : expected_updated_vertices)
    {
        EXPECT_TRUE(vertex_to_check->value_estimate.equals(new_value));
        EXPECT_TRUE(vertex_to_check->value_estimate_for_search.equals(new_value_for_search));
        EXPECT_EQ(vertex_to_check->entropy_estimate, new_entropy_estimate);
    }
    for (shared_ptr<SMVertex> vertex : expected_unchanged_vertices)
    {
        EXPECT_TRUE(vertex->value_estimate.equals(heuristic));
        EXPECT_TRUE(vertex->value_estimate_for_search.equals(heuristic));
        EXPECT_EQ(vertex->entropy_estimate, 0.0);
    }

    // Check num direct updates is as expected (only [[0.25,0.75] directly updated)
    EXPECT_EQ(v_u0->num_direct_updates, 0);
    EXPECT_EQ(v_u1->num_direct_updates, 1);
    EXPECT_EQ(v_u2->num_direct_updates, 0);
    EXPECT_EQ(v_u3->num_direct_updates, 0);
    EXPECT_EQ(v_z0->num_direct_updates, 0);
    EXPECT_EQ(v_z1->num_direct_updates, 0);
    EXPECT_EQ(v_z2->num_direct_updates, 0);

    // Then, each created vertex gets +1 update from creation
    // And V_uX gets +1 update from message passing
    EXPECT_EQ(v_u0->num_updates, 1);
    EXPECT_EQ(v_u1->num_updates, 2);
    EXPECT_EQ(v_u2->num_updates, 2);
    EXPECT_EQ(v_u3->num_updates, 2);
    EXPECT_EQ(v_z0->num_updates, 1);
    EXPECT_EQ(v_z1->num_updates, 1);
    EXPECT_EQ(v_z2->num_updates, 0);
}




/**
Helper to construct a 3D mesh used for testing

                             A
                            /|\
                           / | \
                          /  |  \
                         /   |   \
                        /   F|____\E
                       /     |\  / \
                      /    H | \/   \
                     /       | /G    \
                    /________|/_______\
                    B       D         C

Hijack the construction so we know the exact coordinates
A = [1,0,0]
B = [0,1,0]
C = [0,0,1]

D = 0.5 * [0,1,1]
E = 0.5 * [1,0,1]
F = 0.25 * [2,1,1]
G = 0.25 * [1,1,2]

0.5 * (F+D) = 0.125 * [3,2,3]

H is a point close to the edge DF, whose closest vertex is G

dist(F,D) = ||F-D|| = ||0.25 * [2,-1,-1]|| = sqrt(6)/4 = 0.61ish

N.B. dist(A,B) = sqrt(2) != 1

dist(0.5*(F+D), G) 
 = || 0.125 * [3,2,3] - 0.125 * [2,2,4] ||
 = 0.125 * || [1,0,1] ||
 = sqrt(2)/8
 = 0.18ish

so if we let eps = 0.05, and set H = (1-eps)(0.5 * (F+D)) + eps * B
then dist(H,G) < 0.23ish < 0.3 < 0.61ish / 2 = dist(F,D) / 2

so H is closer to G than F or D

Some other stats we're going to sanity check:
- every simplex is adjacent to 3 edges
- every edge is adjacent to 2 simplices]
- UNLESS, it is on the boundary of the simplex map, when it is adjacent to 1 simplex
- How many vertices are in the simplex map
- How many edges are in the simplex map mesh graph
- How many simplices are in the simplex map mesh graph
- How many non-conforming simplices there are and how many at each depth of binary tree

Example stats for this mesh are:
- 7 vertices (A,B,C,D,E,F,G)
- 11 edges (AB,BD,DF,AF,AE,EF,FG,EG,DG,CD,CE) 
- 5 simplices (ABD,CDE,AEF,DFG,EFG)
- 2 non-conforming simplices (ABD,CDE)
- 1 non-conforming simplex at depth 0
- 1 non-conforming simplex at depth 1
*/

shared_ptr<SMMesh> construct_3d_mesh_for_tests(
    bool find_exact_closest_vertex=true,
    bool eventually_conforming_mesh=true,
    bool always_allow_non_conforming_simplex_to_split=true)
{
    shared_ptr<SMMesh> mesh = make_shared<SMMesh>(
        3, 
        find_exact_closest_vertex, 
        eventually_conforming_mesh, 
        always_allow_non_conforming_simplex_to_split);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh->initialise_mesh(heuristic);

    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);
    Vec E = 0.5 * (A + C);
    Vec F = 0.5 * (A + D);
    Vec G = 0.5 * (D + E);

    shared_ptr<SMVertex> vA = mesh->registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh->registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh->registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh->registry.get_or_create_vertex(D, heuristic, 0.0);
    shared_ptr<SMVertex> vE = mesh->registry.get_or_create_vertex(E, heuristic, 0.0);
    shared_ptr<SMVertex> vF = mesh->registry.get_or_create_vertex(F, heuristic, 0.0);
    shared_ptr<SMVertex> vG = mesh->registry.get_or_create_vertex(G, heuristic, 0.0);

    // Check root vertices are as expected
    vector<shared_ptr<SMVertex>> root_vertices = { vA, vB, vC };
    for (shared_ptr<SMVertex> vertex : root_vertices) 
    {
        if (!mesh->root_simplex->contains_vertex(vertex))
        {
            throw runtime_error("Error in constructing 3d mesh for tests");
        }
    }
    if (mesh->root_simplex->vertices_set.size() != 3 || mesh->root_simplex->vertices_set.size() != 3)
    {
        throw runtime_error("Error in constructing 3d mesh for tests");
    }

    // override longest edge + split
    mesh->root_simplex->longest_edge = std::make_pair(vB, vC);
    mesh->root_simplex->radius = B.dist(C);
    mesh->subdivide_simplex_test(mesh->root_simplex, 0.01, 10, 2);

    // Get child
    shared_ptr<SMSimplex> simplex_ACD = mesh->root_simplex->normal_child;

    // checks 
    vector<shared_ptr<SMVertex>> ACD_vertices = { vA, vC, vD };
    for (shared_ptr<SMVertex> vertex : ACD_vertices) 
    {
        if (!simplex_ACD->contains_vertex(vertex))
        {
            cout << "Expected vertices at:";
            for (shared_ptr<SMVertex> vertex : ACD_vertices) 
            {
                cout << vertex->weight << endl;
            }
            cout << "But got: " << endl;
            for (shared_ptr<SMVertex> vertex : simplex_ACD->vertices) 
            {
                cout << vertex->weight << endl;
            }
            throw runtime_error("Error in constructing 3d mesh for tests");
        }
    }
    if (simplex_ACD->vertices_set.size() != 3 || simplex_ACD->vertices_set.size() != 3)
    {
        throw runtime_error("Error in constructing 3d mesh for tests");
    }

    // override longest edge + split
    simplex_ACD->longest_edge = std::make_pair(vC, vA);
    simplex_ACD->radius = A.dist(C);
    mesh->subdivide_simplex_test(simplex_ACD, 0.01, 10, 2);

    // Get child
    shared_ptr<SMSimplex> simplex_ADE = simplex_ACD->normal_child;

    // checks
    vector<shared_ptr<SMVertex>> ADE_vertices = { vA, vD, vE };
    for (shared_ptr<SMVertex> vertex : ADE_vertices) 
    {
        if (!simplex_ADE->contains_vertex(vertex))
        {
            cout << "Expected vertices at:";
            for (shared_ptr<SMVertex> vertex : ADE_vertices) 
            {
                cout << vertex->weight << endl;
            }
            cout << "But got: " << endl;
            for (shared_ptr<SMVertex> vertex : simplex_ADE->vertices) 
            {
                cout << vertex->weight << endl;
            }
            throw runtime_error("Error in constructing 3d mesh for tests");
        }
    }
    if (simplex_ADE->vertices_set.size() != 3 || simplex_ADE->vertices_set.size() != 3)
    {
        throw runtime_error("Error in constructing 3d mesh for tests");
    }

    // Override longest edge of CDE to be CD so we get behaviour we expect for tests using this simplex
    shared_ptr<SMSimplex> simplex_CDE = simplex_ACD->opposite_child;
    simplex_CDE->longest_edge = std::make_pair(vC, vD);
    simplex_CDE->radius = C.dist(D);

    // override longest edge + split
    simplex_ADE->longest_edge = std::make_pair(vA, vD);
    simplex_ADE->radius = A.dist(D);
    mesh->subdivide_simplex_test(simplex_ADE, 0.01, 10, 2);

    // Get child
    shared_ptr<SMSimplex> simplex_DEF = simplex_ADE->normal_child;

    // checks
    vector<shared_ptr<SMVertex>> DEF_vertices = { vD, vE, vF };
    for (shared_ptr<SMVertex> vertex : DEF_vertices) 
    {
        if (!simplex_DEF->contains_vertex(vertex))
        {
            cout << "Expected vertices at:";
            for (shared_ptr<SMVertex> vertex : DEF_vertices) 
            {
                cout << vertex->weight << endl;
            }
            cout << "But got: " << endl;
            for (shared_ptr<SMVertex> vertex : simplex_DEF->vertices) 
            {
                cout << vertex->weight << endl;
            }
            throw runtime_error("Error in constructing 3d mesh for tests");
        }
    }
    if (simplex_DEF->vertices_set.size() != 3 || simplex_DEF->vertices_set.size() != 3)
    {
        throw runtime_error("Error in constructing 3d mesh for tests");
    }

    // override longest edge + split
    simplex_DEF->longest_edge = std::make_pair(vE, vD);
    simplex_DEF->radius = E.dist(D);
    mesh->subdivide_simplex_test(simplex_DEF, 0.01, 10, 2);

    // Get child
    shared_ptr<SMSimplex> simplex_DFG = simplex_DEF->normal_child;

    // checks
    vector<shared_ptr<SMVertex>> DFG_vertices = { vD, vF, vG };
    for (shared_ptr<SMVertex> vertex : DFG_vertices) 
    {
        if (!simplex_DFG->contains_vertex(vertex))
        {
            throw runtime_error("Error in constructing 3d mesh for tests");
        }
    }
    if (simplex_DFG->vertices_set.size() != 3 || simplex_DFG->vertices_set.size() != 3)
    {
        throw runtime_error("Error in constructing 3d mesh for tests");
    }

    // Done!
    return mesh;
}

/**
Check that the number of vertices, edges, simplices, and non-conforming simplices are as expected
*/
void check_mesh_stats_as_expected(
    SMMesh& mesh,
    size_t expected_num_vertices,
    size_t expected_num_edges,
    size_t expected_num_simplices,
    size_t expected_num_non_conforming_simplices,
    unordered_map<int, size_t> expected_num_non_conforming_simplices_by_depth
)
{
    EXPECT_EQ(mesh.all_vertices_vector.size(), expected_num_vertices);
    EXPECT_EQ(mesh.all_vertices_set.size(), expected_num_vertices);
    EXPECT_EQ(mesh.edge_to_simplex_map.size(), expected_num_edges);
    EXPECT_EQ(mesh.simplex_to_edge_map.size(), expected_num_simplices);
    EXPECT_EQ(mesh.non_conforming_simplices.size(), expected_num_non_conforming_simplices);
    EXPECT_EQ(mesh.non_conforming_simplices_by_depth.size(), expected_num_non_conforming_simplices_by_depth.size());
    for (auto [depth, num_non_conforming_simplices] : expected_num_non_conforming_simplices_by_depth)
    {
        EXPECT_EQ(mesh.non_conforming_simplices_by_depth[depth].size(), num_non_conforming_simplices);
    }
}

/**
Helper to check adjacency maps of a mesh
*/
bool edge_is_on_outer_edge_of_simplex_map(SMEdge& edge)
{
    Vec& v0 = edge.v0->weight;
    Vec& v1 = edge.v1->weight;
    for (int i=0; i<3; i++)
    {
        if (v0[i] == 0.0 && v1[i] == 0.0)
        {
            return true;
        }
    }
    return false;
}

void check_adjacency_maps_3d(SMMesh& mesh)
{
    // Each simplex should have 3 edges, IFF it is a conforming simplex
    // Otherwise it should have greater than 3 edges
    for (auto [simplex, adjacent_edges] : mesh.simplex_to_edge_map)
    {
        if (simplex->is_non_conforming)
        {
            EXPECT_GT(adjacent_edges.size(), 3u);
        }
        else
        {
            EXPECT_EQ(adjacent_edges.size(), 3u);
        }
    }

    // Each edge should be adjacent to 2 simplices, unless it is on an outer edge
    for (auto [edge, adjacent_simplices] : mesh.edge_to_simplex_map)
    {
        if (edge_is_on_outer_edge_of_simplex_map(*edge))
        {
            EXPECT_EQ(adjacent_simplices.size(), 1u);
        }
        else
        {
            EXPECT_EQ(adjacent_simplices.size(), 2u);
        }
    }
}

// First check that the unit simplex is as expected
TEST(SimplexMap_SMMesh3D, unit_simplex_is_as_expected)
{
    shared_ptr<SMMesh> mesh = make_shared<SMMesh>(3, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh->initialise_mesh(heuristic);

    check_mesh_stats_as_expected(
        *mesh, 
        3u, // vertices
        3u, // edges
        1u, // simplices
        0u, // non-conforming simplices
        unordered_map<int, size_t>()
    );
    check_adjacency_maps_3d(*mesh);
}

// Check that some splits lead to expected stats
TEST(SimplexMap_SMMesh3D, some_splits_lead_to_expected_stats)
{
    shared_ptr<SMMesh> mesh = make_shared<SMMesh>(3, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh->initialise_mesh(heuristic);

    mesh->subdivide_simplex_test(mesh->root_simplex, 0.01, 10, 2);

    check_mesh_stats_as_expected(
        *mesh, 
        4u, // vertices
        5u, // edges
        2u, // simplices
        0u, // non-conforming simplices
        unordered_map<int, size_t>()
    );
    check_adjacency_maps_3d(*mesh);
}

// Check that the base simplex for other tests is as expected
TEST(SimplexMap_SMMesh3D, base_3d_test_simplex_is_as_expected)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests();

    unordered_map<int, size_t> expected_num_non_conforming_simplices_by_depth = {
        {1, 1},
        {2, 1}
    };
    check_mesh_stats_as_expected(
        *mesh, 
        7, // vertices
        11, // edges
        5, // simplices
        2, // non-conforming simplices
        expected_num_non_conforming_simplices_by_depth);

    check_adjacency_maps_3d(*mesh);
}

// Check lookup of closest vertex in 3D
TEST(SimplexMap_SMMesh3D, lookup_closest_vertex_3d_exact)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests(true);

    // relevant vertices of simplex map:
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);
    Vec E = 0.5 * (A + C);
    Vec F = 0.5 * (A + D);
    Vec G = 0.5 * (D + E);

    shared_ptr<SMVertex> vA = mesh->registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh->registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh->registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh->registry.get_or_create_vertex(D, heuristic, 0.0);
    shared_ptr<SMVertex> vE = mesh->registry.get_or_create_vertex(E, heuristic, 0.0);
    shared_ptr<SMVertex> vF = mesh->registry.get_or_create_vertex(F, heuristic, 0.0);
    shared_ptr<SMVertex> vG = mesh->registry.get_or_create_vertex(G, heuristic, 0.0);

    // Get DFG simplex
    shared_ptr<SMSimplex> simplex_ACD = mesh->root_simplex->normal_child;
    shared_ptr<SMSimplex> simplex_ADE = simplex_ACD->normal_child;
    shared_ptr<SMSimplex> simplex_DEF = simplex_ADE->normal_child;
    shared_ptr<SMSimplex> simplex_DFG = simplex_DEF->normal_child;

    // Compute the point H we want to look up
    double eps = 0.05;
    Vec H = (1-eps) * (0.5 * (F+D)) + eps * B;

    // Check that we get vG from simplex_DFG as expected
    SMVertexSMSimplexPair closest_vertex_and_simplex = mesh->get_closest_vertex_and_adjoining_simplex(H);

    EXPECT_EQ(closest_vertex_and_simplex.first->weight, vG->weight);
    EXPECT_EQ(closest_vertex_and_simplex.first, vG);
    EXPECT_EQ(closest_vertex_and_simplex.second, simplex_DFG);
}



// Check lookup of closest vertex in 3D when not searching for exact closest in local neighbourhood
TEST(SimplexMap_SMMesh3D, lookup_closest_vertex_3d_approx)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests(false);

    // relevant vertices of simplex map:
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);
    Vec E = 0.5 * (A + C);
    Vec F = 0.5 * (A + D);
    Vec G = 0.5 * (D + E);

    shared_ptr<SMVertex> vA = mesh->registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh->registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh->registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh->registry.get_or_create_vertex(D, heuristic, 0.0);
    shared_ptr<SMVertex> vE = mesh->registry.get_or_create_vertex(E, heuristic, 0.0);
    shared_ptr<SMVertex> vF = mesh->registry.get_or_create_vertex(F, heuristic, 0.0);
    shared_ptr<SMVertex> vG = mesh->registry.get_or_create_vertex(G, heuristic, 0.0);

    // Get DFG simplex
    shared_ptr<SMSimplex> simplex_ABD = mesh->root_simplex->opposite_child;

    // Compute the point H we want to look up
    double eps = 0.05;
    Vec H = (1-eps) * (0.5 * (F+D)) + eps * B;

    // Check that we get vD from simplex_ABD as expected
    SMVertexSMSimplexPair closest_vertex_and_simplex = mesh->get_closest_vertex_and_adjoining_simplex(H);

    EXPECT_EQ(closest_vertex_and_simplex.first->weight, vD->weight);
    EXPECT_EQ(closest_vertex_and_simplex.first, vD);
    EXPECT_EQ(closest_vertex_and_simplex.second, simplex_ABD);
}

// Test that calling maybe subdivide with existing non-conforming 
// This also checks that a new simplex can be created that is already non-conforming because we split many times along 
//      the opposite side of binary tree
TEST(SimplexMap_SMMesh3D, maybe_subdivide_conforming_simplex_with_non_conforming_simplex_in_mesh)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests(true);

    // relevant vertices of simplex map:
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);
    Vec E = 0.5 * (A + C);
    Vec F = 0.5 * (A + D);
    Vec G = 0.5 * (D + E);

    shared_ptr<SMVertex> vA = mesh->registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh->registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh->registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh->registry.get_or_create_vertex(D, heuristic, 0.0);
    shared_ptr<SMVertex> vE = mesh->registry.get_or_create_vertex(E, heuristic, 0.0);
    shared_ptr<SMVertex> vF = mesh->registry.get_or_create_vertex(F, heuristic, 0.0);
    shared_ptr<SMVertex> vG = mesh->registry.get_or_create_vertex(G, heuristic, 0.0);

    // Get non-conforming simplex at depth 1
    shared_ptr<SMSimplex> simplex_ABD = mesh->root_simplex->opposite_child;

    // Get a CDE simplex at depth 2
    shared_ptr<SMSimplex> simplex_AEF = mesh->root_simplex->normal_child->normal_child->opposite_child;

    // Update the value estimate of vE (manually) so that AEF it passes checks
    vE->value_estimate = make_vec(1.0, 1.0, 1.0);

    // Assert that AEF is conforming
    EXPECT_FALSE(simplex_AEF->is_non_conforming);

    // Call maybe subdivide
    mesh->maybe_subdivide(simplex_AEF, 0.01, 10, 2);

    // Check that split_counter was incremented
    EXPECT_EQ(simplex_AEF->split_counter, 1);

    // Check that simplex_AEF was not subdivided, and ABD was
    EXPECT_EQ(simplex_AEF->normal_child, nullptr);
    EXPECT_EQ(simplex_AEF->opposite_child, nullptr);
    EXPECT_NE(simplex_ABD->normal_child, nullptr);
    EXPECT_NE(simplex_ABD->opposite_child, nullptr);

    // And check the stats are as expected, now that edge AB should have been split
    unordered_map<int, size_t> expected_num_non_conforming_simplices_by_depth = {
        {2, 2}
    };
    check_mesh_stats_as_expected(
        *mesh, 
        8, // vertices
        13, // edges
        6, // simplices
        2, // non-conforming simplices (expect a new non-conforming simplex at depth 2)
        expected_num_non_conforming_simplices_by_depth
    );
}

// Test that calling maybe subdivide on non-conforming simplex will split it, even if threshold stuff is not set, 
// and it will split another non-conforming simplex if it exists
TEST(SimplexMap_SMMesh3D, maybe_subdivide_non_conforming_simplex)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests(true);

    // relevant vertices of simplex map:
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);
    Vec E = 0.5 * (A + C);
    Vec F = 0.5 * (A + D);
    Vec G = 0.5 * (D + E);

    shared_ptr<SMVertex> vA = mesh->registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh->registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh->registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh->registry.get_or_create_vertex(D, heuristic, 0.0);
    shared_ptr<SMVertex> vE = mesh->registry.get_or_create_vertex(E, heuristic, 0.0);
    shared_ptr<SMVertex> vF = mesh->registry.get_or_create_vertex(F, heuristic, 0.0);
    shared_ptr<SMVertex> vG = mesh->registry.get_or_create_vertex(G, heuristic, 0.0);

    // Get non-conforming simplex at depth 1
    shared_ptr<SMSimplex> simplex_ABD = mesh->root_simplex->opposite_child;

    // Get a CDE simplex at depth 2
    shared_ptr<SMSimplex> simplex_CDE = mesh->root_simplex->normal_child->opposite_child;

    // Assert that CDE is non conforming
    EXPECT_TRUE(simplex_CDE->is_non_conforming);

    // Call maybe subdivide
    mesh->maybe_subdivide(simplex_CDE, 0.01, 10, 2);

    // Check that both simplices were subdivided
    EXPECT_NE(simplex_CDE->normal_child, nullptr);
    EXPECT_NE(simplex_CDE->opposite_child, nullptr);
    EXPECT_NE(simplex_ABD->normal_child, nullptr);
    EXPECT_NE(simplex_ABD->opposite_child, nullptr);

    // And check the stats are as expected, now that edge AB should have been split
    unordered_map<int, size_t> expected_num_non_conforming_simplices_by_depth = {
        {2, 1},
        {3, 1}
    };
    check_mesh_stats_as_expected(
        *mesh, 
        9, // vertices
        15, // edges
        7, // simplices
        2, // non-conforming simplices
        expected_num_non_conforming_simplices_by_depth
    );
}

// Test that maybe subdivide gracefully handles when it is called on the one non-conforming simplex in the map
// Worth noting that the constructed simplex has 2 non conforming simplices
// But when each are split, they create another non-conforming simplex
// So if we pop a non-conforming simplex and subdivide it until the mesh is conforming, 
// then we would have to do it 4 times
TEST(SimplexMap_SMMesh3D, maybe_subdivide_with_one_non_conforming_simplex)
{
    shared_ptr<SMMesh> mesh = construct_3d_mesh_for_tests(true);

    // subdivide the first 3 non-conforming simplices manually
    shared_ptr<SMSimplex> non_conforming_simplex_1 = mesh->pop_lowest_depth_non_conforming_simplex_test();
    mesh->subdivide_simplex_test(non_conforming_simplex_1, 0.01, 10, 2);
    shared_ptr<SMSimplex> non_conforming_simplex_2 = mesh->pop_lowest_depth_non_conforming_simplex_test();
    mesh->subdivide_simplex_test(non_conforming_simplex_2, 0.01, 10, 2);
    shared_ptr<SMSimplex> non_conforming_simplex_3 = mesh->pop_lowest_depth_non_conforming_simplex_test();
    mesh->subdivide_simplex_test(non_conforming_simplex_3, 0.01, 10, 2);

    // Check stats are as expected before final split
    unordered_map<int, size_t> expected_num_non_conforming_simplices_by_depth = {
        {3, 1}
    };
    check_mesh_stats_as_expected(
        *mesh, 
        9, // vertices
        16, // edges
        8, // simplices
        1, // non-conforming simplices
        expected_num_non_conforming_simplices_by_depth
    );

    // For the final non-conforming simplex, call maybe subdivide
    shared_ptr<SMSimplex> non_conforming_simplex_4 = mesh->pop_lowest_depth_non_conforming_simplex_test();
    mesh->maybe_subdivide(non_conforming_simplex_4, 0.01, 10, 2);

    // Assert that there are no non-conforming simplices left
    EXPECT_TRUE(mesh->non_conforming_simplices.empty());
    EXPECT_TRUE(mesh->non_conforming_simplices_by_depth.empty());

    // And check the stats are as expected
    check_mesh_stats_as_expected(
        *mesh, 
        9, // vertices
        17, // edges
        9, // simplices
        0, // non-conforming simplices
        unordered_map<int, size_t>()
    );
}

// Check neighbours updated correctly, with one split on a 3D mesh
// I.e. test that new edges created get connected
TEST(SimplexMap_SMMesh3D, check_neighbours_updated_correctly_with_one_split)
{
    SMMesh mesh(3, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh.initialise_mesh(heuristic);

    Vec A = make_vec(1.0, 0.0, 0.0);
    Vec B = make_vec(0.0, 1.0, 0.0);
    Vec C = make_vec(0.0, 0.0, 1.0);
    Vec D = 0.5 * (B + C);

    shared_ptr<SMVertex> vA = mesh.registry.get_or_create_vertex(A, heuristic, 0.0);
    shared_ptr<SMVertex> vB = mesh.registry.get_or_create_vertex(B, heuristic, 0.0);
    shared_ptr<SMVertex> vC = mesh.registry.get_or_create_vertex(C, heuristic, 0.0);
    shared_ptr<SMVertex> vD = mesh.registry.get_or_create_vertex(D, heuristic, 0.0);

    // override longest edge + split
    mesh.root_simplex->longest_edge = std::make_pair(vB, vC);
    mesh.root_simplex->radius = B.dist(C);
    mesh.subdivide_simplex_test(mesh.root_simplex, 0.01, 10, 2);

    // Check number of neighbours is as expected
    EXPECT_EQ(vA->neighbours->size(), 3u);
    EXPECT_EQ(vB->neighbours->size(), 2u);
    EXPECT_EQ(vC->neighbours->size(), 2u);
    EXPECT_EQ(vD->neighbours->size(), 3u);
}

// Finally, test message passing on a 3D mesh that we fully subdivided to a depth of 4
TEST(SimplexMap_SMMesh3D_cur, message_passing_on_fully_subdivided_3d_mesh)
{
    SMMesh mesh(3, true, true, true);
    Vec heuristic = make_vec(0.0, 0.0, 0.0);
    mesh.initialise_mesh(heuristic);

    // Subdivide to depth of 4
    // Make depth 1
    mesh.subdivide_simplex_test(mesh.root_simplex, 0.01, 10, 2);

    shared_ptr<SMSimplex> l = mesh.root_simplex->opposite_child;
    shared_ptr<SMSimplex> r = mesh.root_simplex->normal_child;

    // Make depth 2
    mesh.subdivide_simplex_test(l, 0.01, 10, 2);
    mesh.subdivide_simplex_test(r, 0.01, 10, 2);

    shared_ptr<SMSimplex> ll = l->opposite_child;
    shared_ptr<SMSimplex> lr = l->normal_child;
    shared_ptr<SMSimplex> rl = r->opposite_child;
    shared_ptr<SMSimplex> rr = r->normal_child;

    // Make depth 3
    mesh.subdivide_simplex_test(ll, 0.01, 10, 2);
    mesh.subdivide_simplex_test(lr, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rl, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rr, 0.01, 10, 2);

    // Make depth 4
    shared_ptr<SMSimplex> lll = ll->opposite_child;
    shared_ptr<SMSimplex> llr = ll->normal_child;
    shared_ptr<SMSimplex> lrl = lr->opposite_child;
    shared_ptr<SMSimplex> lrr = lr->normal_child;
    shared_ptr<SMSimplex> rll = rl->opposite_child;
    shared_ptr<SMSimplex> rlr = rl->normal_child;
    shared_ptr<SMSimplex> rrl = rr->opposite_child;
    shared_ptr<SMSimplex> rrr = rr->normal_child;

    mesh.subdivide_simplex_test(lll, 0.01, 10, 2);
    mesh.subdivide_simplex_test(llr, 0.01, 10, 2);
    mesh.subdivide_simplex_test(lrl, 0.01, 10, 2);
    mesh.subdivide_simplex_test(lrr, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rll, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rlr, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rrl, 0.01, 10, 2);
    mesh.subdivide_simplex_test(rrr, 0.01, 10, 2);

    // Check stats are as expected
    check_mesh_stats_as_expected(
        mesh, 
        15, // vertices
        30, // edges
        16, // simplices
        0, // non-conforming simplices
        unordered_map<int, size_t>()
    );
    
    // Get vertex at corner of simplex)
    // Calling this A because it sort of corresponds to vertex A in the example mesh we gave
    // But because when all edges are same length, just pick the first one to split
    // So usually (1,0,0) <-> (0,1,0) is chosen to split first
    // Diagrams we made to work things out assume A is opposite vertex to split edge at (0,0,1)
    // But we can just use (0,0,1) instead by rotational symmetry and it should work out the same
    Vec A = make_vec(0.0, 0.0, 1.0);
    shared_ptr<SMVertex> vA = mesh.registry.get_or_create_vertex(A, heuristic, 0.0);

    // Manually set value estimate for vA
    Vec value_estimate = make_vec(1.0, 1.0, 1.0);
    Vec value_estimate_for_search = make_vec(1.0, 1.0, 1.0);
    double entropy_estimate = 1.0;

    // Call message passing from vA
    RandManager rand_manager(42);
    mesh.update_vertex_values_and_share(
        rand_manager, 
        vA, 
        2, //max push radius = 2
        -1, 
        value_estimate, 
        value_estimate_for_search, 
        entropy_estimate);

    // There should be 3 1-hop neighbours of vA
    // And 5 2-hop neighbours of vA (excluding 1-hop neighbours)
    // So there should be 1+3++5=9 values updated by this call
    int num_updated_vertices = 0;
    for (shared_ptr<SMVertex> vertex : mesh.all_vertices_set)
    {
        if (vertex->value_estimate.equals(value_estimate))
        {
            num_updated_vertices++;
        }
    }
    EXPECT_EQ(num_updated_vertices, 9);
}
