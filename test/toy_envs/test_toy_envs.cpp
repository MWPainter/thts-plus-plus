#include "test/toy_envs/test_toy_envs.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "toy_envs/frozen_lake_env.h"
#include "toy_envs/d_chain_env.h"
#include "toy_envs/sailing_env.h"

// includes
#include "thts_manager.h"
#include "helper_templates.h"

#include <algorithm>
#include <string>
#include <unordered_map>


using namespace std;
using namespace thts;

#include <iostream>


/**
 * Just runs through a bunch of sequences of actions
 */
static const std::string FL_TEST_MAP[] =
    {"SFF",
     "FHF",
     "FFG"};

TEST(ToyEnvs_TestFrozenLake, sanity_check) 
{   
    shared_ptr<const IntAction> right_act = make_shared<IntAction>(FL_RIGHT);
    shared_ptr<const IntAction> down_act = make_shared<IntAction>(FL_DOWN);
    shared_ptr<const IntAction> left_act = make_shared<IntAction>(FL_LEFT);
    shared_ptr<const IntAction> up_act = make_shared<IntAction>(FL_UP);

    FrozenLakeEnv env(3,3,FL_TEST_MAP);

    vector<shared_ptr<const IntAction>> acts_1 = 
        {right_act,
         down_act};
    vector<shared_ptr<const IntAction>> acts_2 = 
        {left_act,
         left_act,
         right_act,
         up_act,
         right_act,
         up_act,
         down_act,
         right_act,
         down_act};
    vector<shared_ptr<const IntAction>> acts_3 = 
        {right_act,
         right_act,
         left_act,
         right_act,
         down_act,
         up_act,
         down_act,
         down_act};
    vector<shared_ptr<const IntAction>> acts_4 = 
        {up_act,
         up_act,
         down_act,
         left_act,
         down_act,
         left_act,
         right_act,
         down_act,
         right_act};
    vector<shared_ptr<const IntAction>> acts_5 = 
        {down_act,
         down_act,
         up_act,
         down_act,
         right_act,
         left_act,
         right_act,
         right_act};
    vector<vector<shared_ptr<const IntAction>>> actss = {acts_1, acts_2, acts_3, acts_4, acts_5};

    for (u_long j=0; j < actss.size(); j++) {
        shared_ptr<const Int3TupleState> state = env.get_initial_state();
        for (u_long i=0; i < actss[j].size(); i++) {
            EXPECT_FALSE(env.is_sink_state(state));
            shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
            EXPECT_EQ(valid_actions->size(), 4u);
            double rew = env.get_reward(state, actss[j][i]);
            // first seq goes in hole, rest reach goal
            double expected_rew = (j == 0u || i < actss[j].size() - 1) ? 0.0 : 1.0;
            EXPECT_EQ(rew, expected_rew);
            state = env.sample_transition_distribution(state, actss[j][i]);
        }
        EXPECT_TRUE(env.is_sink_state(state));
        shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
        EXPECT_EQ(valid_actions->size(), 0u);
    }
}

TEST(ToyEnvs_TestDChain, sanity_check) 
{   
    shared_ptr<const IntAction> right_act = make_shared<IntAction>(DCHAIN_RIGHT);
    shared_ptr<const IntAction> down_act = make_shared<IntAction>(DCHAIN_DOWN);

    DChainEnv env(5, 2.0);

    // test paths with down
    for (int i=0; i<5; i++) {
        shared_ptr<const IntState> state = env.get_initial_state();
        for (int j=0; j<i; j++) {
            shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
            EXPECT_EQ(valid_actions->size(), 2u);

            double rew = env.get_reward(state, right_act);
            EXPECT_EQ(rew, 0.0);

            state = env.sample_transition_distribution(state, right_act);
            EXPECT_FALSE(env.is_sink_state(state));
        }
        double rew = env.get_reward(state, down_act);
        EXPECT_EQ(rew, (5-i-1)/5.0);
        state = env.sample_transition_distribution(state, down_act);
        EXPECT_TRUE(env.is_sink_state(state));
    }

    // test getting final reward
    shared_ptr<const IntState> state = env.get_initial_state();
    for (int j=0; j<5; j++) {
        EXPECT_FALSE(env.is_sink_state(state));
        shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
        EXPECT_EQ(valid_actions->size(), 2u);
        double rew = env.get_reward(state, right_act);
        // expect reward on last iter
        double expected_rew = (j==4) ? 2.0 : 0.0;
        EXPECT_EQ(rew, expected_rew);
        state = env.sample_transition_distribution(state, right_act);
    }
    EXPECT_TRUE(env.is_sink_state(state));
    shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
    EXPECT_EQ(valid_actions->size(), 0u);
}

bool check_valid_action_helper(shared_ptr<IntActionVector> valid_actions, shared_ptr<const IntAction> action) {
    for (shared_ptr<const IntAction> valid_action : *valid_actions) {
        if (valid_action->action == action->action) {
            return true;
        }
    }
    return false;
}

TEST(ToyEnvs_TestSailing, sanity_check) 
{   
    shared_ptr<const IntAction> ne_act = make_shared<IntAction>(NE);
    shared_ptr<const IntAction> nn_act = make_shared<IntAction>(NN);
    shared_ptr<const IntAction> ee_act = make_shared<IntAction>(EE);
    shared_ptr<const IntAction> se_act = make_shared<IntAction>(SE);
    shared_ptr<const IntAction> nw_act = make_shared<IntAction>(NW);
    IntActionVector acts_to_try = {ne_act, nn_act, ee_act, se_act, nw_act};

    SailingEnv env(10, 10);
    RandManager rand_manager;

    // use sw as init wind direction
    shared_ptr<const Int3TupleState> init_state = make_shared<const Int3TupleState>(0,0,SW);

    // check only two actions
    EXPECT_EQ(env.get_valid_actions(init_state)->size(), 2u);
    cout << helper::vector_pretty_print_string(*env.get_valid_actions(init_state)) << endl;

    shared_ptr<const Int3TupleState> state = init_state;
    while (!env.is_sink_state(state)) {
        shared_ptr<IntActionVector> valid_actions = env.get_valid_actions(state);
        shared_ptr<const IntAction> action = nullptr;
        for (shared_ptr<const IntAction> act : acts_to_try) {
            if (check_valid_action_helper(valid_actions, act)) {
                action = act;
                break;
            }
        }
        if (action == nullptr) {
            FAIL();
            exit(60415);
        }

        if (get<0>(state->state) < 0 || get<0>(state->state) > 10 || get<1>(state->state) < 0 || get<1>(state->state) > 10) {
            FAIL();
            exit(12);
        }

        double rew = env.get_reward(state, action);
        cout << helper::vector_pretty_print_string(*valid_actions) << endl;
        cout << state << "," << action << "," << rew << endl;
        state = env.sample_transition_distribution(state, action, rand_manager);
    }
}