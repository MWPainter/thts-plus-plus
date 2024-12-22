#include "test/mo/test_simplex_map.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "mo/simplex_map.h"

// includes
#include <string>
#include <utility>

#include <Eigen/Dense>

#include "mo/mo_helper.h"

#include "mo/mo_thts_context.h"

#include "test/mo/test_mo_thts_env.h"


using namespace std;
using namespace thts;
using namespace thts::test;





static Eigen::ArrayXd make_3vec(double a, double b, double c) {
    Eigen::ArrayXd v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return v;
}

static Eigen::ArrayXd make_4vec(double a, double b, double c, double d) {
    Eigen::ArrayXd v(4);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    return v;
}

static bool approx_eq_vec(Eigen::ArrayXd x, Eigen::ArrayXd y) {
    double eps = 1e-10;
    return ((x-y).abs() < eps).all();
}

static Eigen::ArrayXd normalised(Eigen::ArrayXd x) {
    return x / sqrt((x*x).sum());
}

static bool check_expected_simplex(shared_ptr<vector<shared_ptr<NGV>>> simplex, unordered_set<shared_ptr<NGV>>& expected_simplex) {
    for (shared_ptr<NGV> v : *simplex) {
        if (!expected_simplex.contains(v)) {
            return false;
        }
    }
    return true;
}




// Very hacky method of getting a ThtsEnv with a mocked "get_reward_dim"
class MockMoThtsEnv : public TestMoThtsEnv {
    public:
        MockMoThtsEnv(int rew_dim) : 
            TestMoThtsEnv(2)
        {
            reward_dim = rew_dim;
        } 

        virtual ~MockMoThtsEnv() = default;
};




// /**
//  *  
//  */
// TEST(Sm_Integration, test_subdivisions) {    
//     // Todo - actually work out the correct normals for this
//     // TODO: work out mocking the thts env. Issue is that there is a call to get_reward_dim, which isnt virtual, and I 
//     // only want to make things virtual if they need to be in main lib
//     // Apparenlty there are some other libraries than gmock for this
//     // \Just use the test thts env for now
//     shared_ptr<MockMoThtsEnv> mock_thts_env = make_shared<MockMoThtsEnv>(3);

//     Eigen::ArrayXd zero = Eigen::ArrayXd::Zero(3);

//     SmtThtsManagerArgs manager_args(mock_thts_env, zero);
//     SmtThtsManager manager(manager_args);
//     Eigen::ArrayXd e1 = make_3vec(1,0,0);
//     Eigen::ArrayXd e2 = make_3vec(0,1,0);
//     Eigen::ArrayXd e3 = make_3vec(0,0,1);
//     shared_ptr<NGV> ngv1 = make_shared<NGV>(e1,zero,0);
//     shared_ptr<NGV> ngv2 = make_shared<NGV>(e2,zero,0);
//     shared_ptr<NGV> ngv3 = make_shared<NGV>(e3,zero,0);
//     shared_ptr<NGV> ngv12 = make_shared<NGV>((e1+e2)/2,zero,0);
//     shared_ptr<NGV> ngv13 = make_shared<NGV>((e1+e3)/2,zero,0);
//     shared_ptr<NGV> ngv23 = make_shared<NGV>((e2+e3)/2,zero,0);
//     shared_ptr<NGV> ngv1_12 = make_shared<NGV>((e1+(e1+e2)/2)/2,zero,0);
//     shared_ptr<NGV> ngv2_12 = make_shared<NGV>((e2+(e1+e2)/2)/2,zero,0);
//     shared_ptr<NGV> ngv3_12 = make_shared<NGV>((e3+(e1+e2)/2)/2,zero,0);
//     SimplexMap map(3,Eigen::ArrayXd::Zero(3));

//     // ---

//     shared_ptr<NGV> expected_new_ngv;
//     Eigen::ArrayXd expected_new_normal;
//     unordered_set<shared_ptr<NGV>> expected_simplex;

//     // ---

//     expected_new_ngv = ngv12;
//     expected_new_normal = normalised(make_3vec(1,-1,0));
//     expected_simplex = {ngv1, ngv2, ngv3};
    
//     shared_ptr<TN> depth_0_tn = map.root_node;
//     depth_0_tn->splitting_edge_normal_side_vertex = ngv1;
//     depth_0_tn->splitting_edge_opposite_side_vertex = ngv2;
//     depth_0_tn->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_0_tn->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_0_tn->splitting_edge_new_vertex);
//     EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_0_tn->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv13;
//     expected_new_normal = normalised(make_3vec(1,0,-1));
//     expected_simplex = {ngv1,ngv3,ngv12};
    
//     shared_ptr<TN> depth_1_tn_n = depth_0_tn->normal_side_child;
//     depth_1_tn_n->splitting_edge_normal_side_vertex = ngv1;
//     depth_1_tn_n->splitting_edge_opposite_side_vertex = ngv3;
//     depth_1_tn_n->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_1_tn_n->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_1_tn_n->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_n->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv23;
//     expected_new_normal = normalised(make_3vec(0,1,-1));
//     expected_simplex = {ngv2,ngv3,ngv12};
    
//     shared_ptr<TN> depth_1_tn_o = depth_0_tn->opposite_side_child;
//     depth_1_tn_o->splitting_edge_normal_side_vertex = ngv2;
//     depth_1_tn_o->splitting_edge_opposite_side_vertex = ngv3;
//     depth_1_tn_o->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_1_tn_o->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_1_tn_o->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_o->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv1_12;
//     // expected_new_normal = normalised(make_3vec(0,1,-1));
//     expected_simplex = {ngv1,ngv12,ngv13};

//     shared_ptr<TN> depth_2_tn_nn = depth_1_tn_n->normal_side_child;
//     depth_2_tn_nn->splitting_edge_normal_side_vertex = ngv1;
//     depth_2_tn_nn->splitting_edge_opposite_side_vertex = ngv12;
//     depth_2_tn_nn->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_2_tn_nn->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_2_tn_nn->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_o->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv3_12;
//     // expected_new_normal = normalised(make_3vec(0,1,-1));
//     expected_simplex = {ngv3,ngv12,ngv13};

//     shared_ptr<TN> depth_2_tn_no = depth_1_tn_n->opposite_side_child;
//     depth_2_tn_no->splitting_edge_normal_side_vertex = ngv3;
//     depth_2_tn_no->splitting_edge_opposite_side_vertex = ngv12;
//     depth_2_tn_no->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_2_tn_no->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_2_tn_no->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_o->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv2_12;
//     // expected_new_normal = normalised(make_3vec(0,1,-1));
//     expected_simplex = {ngv2,ngv12,ngv23};

//     shared_ptr<TN> depth_2_tn_on = depth_1_tn_o->normal_side_child;
//     depth_2_tn_on->splitting_edge_normal_side_vertex = ngv2;
//     depth_2_tn_on->splitting_edge_opposite_side_vertex = ngv12;
//     depth_2_tn_on->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_2_tn_on->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_2_tn_on->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_o->splitting_hyperplane_normal));

//     // ---

//     expected_new_ngv = ngv3_12;
//     // expected_new_normal = normalised(make_3vec(0,1,-1));
//     expected_simplex = {ngv3,ngv12,ngv23};

//     shared_ptr<TN> depth_2_tn_oo = depth_1_tn_o->opposite_side_child;
//     depth_2_tn_oo->splitting_edge_normal_side_vertex = ngv3;
//     depth_2_tn_oo->splitting_edge_opposite_side_vertex = ngv12;
//     depth_2_tn_oo->create_children_binary_tree(map);

//     EXPECT_TRUE(check_expected_simplex(depth_2_tn_oo->simplex_vertices, expected_simplex));
//     EXPECT_EQ(expected_new_ngv, depth_2_tn_oo->splitting_edge_new_vertex);
//     // EXPECT_TRUE(approx_eq_vec(expected_new_normal, depth_1_tn_o->splitting_hyperplane_normal));

//     // ---

//     // Count the number of LSE's and NGVs is as expected
//     EXPECT_EQ(map.n_graph_vertex_set->size(), 9u);
//     EXPECT_EQ(map.n_graph_vertices->size(), 9u);

//     unordered_set<shared_ptr<LSE>> lse_set;
//     for (pair<UnorderedNGVPair,shared_ptr<LSE>> pair : map.lse_map) {
//         lse_set.insert(pair.second);
//     }
//     EXPECT_EQ(lse_set.size(), 4u); // n.b. LSE's are lazily created, so only edges which are subdivided are 
// } 




/**
 *  note that this test should never pass as is
 * want something that checks it's doing approx nearest neighbour lookups
 */
TEST(Sm_Integration, test_get_closest_ngv_for_weight) {    
    // Create simplex map (copied from Sm_integration.test_subdivisions)
    SimplexMap map(3,Eigen::ArrayXd::Zero(3));

    shared_ptr<MockMoThtsEnv> mock_thts_env = make_shared<MockMoThtsEnv>(3);
    Eigen::ArrayXd zero = Eigen::ArrayXd::Zero(3);
    SmtThtsManagerArgs manager_args(mock_thts_env, zero);
    manager_args.seed = 60145;
    manager_args.simplex_map_splitting_option = SPLIT_ordered;
    manager_args.simplex_node_l_inf_thresh = std::numeric_limits<double>::min();
    SmtThtsManager manager(manager_args);

    for (int i=0; i<100; i++) {
        shared_ptr<TN> node = map.root_node;
        while (node->has_children()) {
            if (node->l_inf_norm < manager.simplex_node_l_inf_thresh) {
                break;
            }
            if (manager.get_rand_int(0,2) == 1) {
                node = node->normal_side_child;
            } else {
                node = node->opposite_side_child;
            }
        }
        node->create_children(manager);
    }

    // Sample 1000 weights, get the closest NGV, and check that we got the correct NGV
    for (int i=0; i<1000; i++) {
        Eigen::ArrayXd ctx_weight = MoThtsContext::sample_uniform_random_simplex_for_weight(manager);
        shared_ptr<NGV> closest_ngv = map.get_leaf_tn_node(ctx_weight)->get_closest_ngv_vertex(ctx_weight);

        double min_dist = numeric_limits<double>::max();
        shared_ptr<NGV> true_closest_ngv = nullptr;
        for (shared_ptr<NGV> ngv : *map.n_graph_vertices) {
            double dist = thts::helper::dist(ngv->weight,ctx_weight);
            if (dist < min_dist) {
                min_dist = dist;
                true_closest_ngv = ngv;
            }
        }

        EXPECT_EQ(closest_ngv,true_closest_ngv);
        // if (*closest_ngv != *true_closest_ngv) {
        //     cout << "---" << endl 
        //         << "context weight" << endl << ctx_weight << endl << endl
        //         << "weight from map" << endl << closest_ngv->weight << endl << "(dist= " << thts::helper::dist(closest_ngv->weight,ctx_weight) << ")" << endl
        //         << "true closest weight in map" << endl << true_closest_ngv->weight << endl << "(dist= " << thts::helper::dist(true_closest_ngv->weight,ctx_weight) << ")" << endl;
        // }
    }
} 







// TODO: do a 4D simplex test too