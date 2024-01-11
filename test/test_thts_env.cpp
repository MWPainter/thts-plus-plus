#include "test/test_thts_env.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "thts_env.h"

// includes
#include "thts_types.h"

#include "test/test_thts_manager.h"


using namespace std;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for assertations)
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::StrEq;




bool action_vector_contains(StringActionVector& actions, string str) {
    for (shared_ptr<const StringAction> action : actions) {
        if (action->action == str) {
            return true;
        }
    }
    return false;
}

shared_ptr<StringActionVector> convert_action_vector(shared_ptr<ActionVector> actions) {
    shared_ptr<StringActionVector> actions_copy = make_shared<StringActionVector>();
    for (shared_ptr<const Action> act : *actions) {
        actions_copy->push_back(static_pointer_cast<const StringAction>(act));
    }
    return actions_copy;
}



TEST(Env_MdpImplementation, test_interaction_as_expected)
{
    TestThtsEnv env(1);

    shared_ptr<const StringAction> l_act = make_shared<const StringAction>("left");
    shared_ptr<const StringAction> r_act = make_shared<const StringAction>("right");
    shared_ptr<const StringAction> d_act = make_shared<const StringAction>("down");
    shared_ptr<const StringAction> u_act = make_shared<const StringAction>("up");

    // Setup mock manager, testing fully observable env, so expect no calls to it
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);

    // init state
    ThtsEnvContext ctx;
    shared_ptr<const IntPairState> init_state = env.get_initial_state();
    EXPECT_EQ(init_state->state.first, 0);
    EXPECT_EQ(init_state->state.second, 0);

    // actions from init state
    shared_ptr<StringActionVector> actions_ptr = env.get_valid_actions(init_state, ctx);
    StringActionVector& actions = *actions_ptr;
    EXPECT_EQ(actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(actions, "right"));
    EXPECT_TRUE(action_vector_contains(actions, "down"));

    // init state is not sink
    EXPECT_FALSE(env.is_sink_state(init_state,ctx));

    // Move right
    shared_ptr<const IntPairState> r_state = env.sample_transition_distribution(init_state, r_act, *manager_ptr, ctx);
    EXPECT_EQ(r_state->state.first, 1);
    EXPECT_EQ(r_state->state.second, 0);

    // Check reward (cost) = -1
    double r_reward = env.get_reward(init_state, r_act, ctx);
    EXPECT_EQ(r_reward, -1.0);

    // actions from r state
    shared_ptr<StringActionVector> r_actions_ptr = env.get_valid_actions(r_state, ctx);
    StringActionVector& r_actions = *r_actions_ptr;
    EXPECT_EQ(r_actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(r_actions, "left"));
    EXPECT_TRUE(action_vector_contains(r_actions, "down"));

    // r state is not sink
    EXPECT_FALSE(env.is_sink_state(r_state,ctx));

    // Move down
    shared_ptr<const IntPairState> rd_state = env.sample_transition_distribution(r_state, d_act, *manager_ptr, ctx);
    EXPECT_EQ(rd_state->state.first, 1);
    EXPECT_EQ(rd_state->state.second, 1);

    // Check reward (cost) = -1
    double rd_reward = env.get_reward(r_state, d_act, ctx);
    EXPECT_EQ(rd_reward, -1.0);

    // no actions from rd_state as sink
    shared_ptr<StringActionVector> rd_actions_ptr = env.get_valid_actions(rd_state, ctx);
    StringActionVector& rd_actions = *rd_actions_ptr;
    EXPECT_EQ(rd_actions.size(), 0ul);

    // rd state IS sink
    EXPECT_TRUE(env.is_sink_state(rd_state,ctx));


    // Also try move down from init state
    shared_ptr<const IntPairState> d_state = env.sample_transition_distribution(init_state, d_act, *manager_ptr, ctx);
    EXPECT_EQ(d_state->state.first, 0);
    EXPECT_EQ(d_state->state.second, 1);

    // Check reward (cost) = -1
    double d_reward = env.get_reward(init_state, d_act, ctx);
    EXPECT_EQ(d_reward, -1.0);

    // actions from d state
    shared_ptr<StringActionVector> d_actions_ptr = env.get_valid_actions(d_state, ctx);
    StringActionVector& d_actions = *d_actions_ptr;
    EXPECT_EQ(d_actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(d_actions, "right"));
    EXPECT_TRUE(action_vector_contains(d_actions, "up"));

    // r state is not sink
    EXPECT_FALSE(env.is_sink_state(d_state,ctx));
}

TEST(Env_MdpImplementation, test_interface_interaction_as_expected)
{
    TestThtsEnv env(1);

    shared_ptr<const StringAction> l_act = make_shared<const StringAction>("left");
    shared_ptr<const StringAction> r_act = make_shared<const StringAction>("right");
    shared_ptr<const StringAction> d_act = make_shared<const StringAction>("down");
    shared_ptr<const StringAction> u_act = make_shared<const StringAction>("up");

    shared_ptr<const Action> l_act_itfc = static_pointer_cast<const Action>(l_act);
    shared_ptr<const Action> r_act_itfc = static_pointer_cast<const Action>(r_act);
    shared_ptr<const Action> d_act_itfc = static_pointer_cast<const Action>(d_act);
    shared_ptr<const Action> u_act_itfc = static_pointer_cast<const Action>(u_act);

    // Setup mock manager, testing fully observable env, so expect no calls to it
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);

    // init state
    ThtsEnvContext ctx;
    shared_ptr<const State> init_state_itfc = env.get_initial_state_itfc();
    shared_ptr<const IntPairState> init_state = static_pointer_cast<const IntPairState>(init_state_itfc);
    EXPECT_EQ(init_state->state.first, 0);
    EXPECT_EQ(init_state->state.second, 0);

    // actions from init state
    shared_ptr<ActionVector> actions_ptr_itfc = env.get_valid_actions_itfc(init_state_itfc, ctx);
    shared_ptr<StringActionVector> actions_ptr = convert_action_vector(actions_ptr_itfc);
    StringActionVector& actions = *actions_ptr;
    EXPECT_EQ(actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(actions, "right"));
    EXPECT_TRUE(action_vector_contains(actions, "down"));

    // init state is not sink
    EXPECT_FALSE(env.is_sink_state_itfc(init_state_itfc,ctx));

    // Move right
    shared_ptr<const State> r_state_itfc = env.sample_transition_distribution_itfc(
        init_state_itfc, r_act_itfc, *manager_ptr, ctx);
    shared_ptr<const IntPairState> r_state = static_pointer_cast<const IntPairState>(r_state_itfc);
    EXPECT_EQ(r_state->state.first, 1);
    EXPECT_EQ(r_state->state.second, 0);

    // Check reward (cost) = -1
    double r_reward = env.get_reward_itfc(init_state_itfc, r_act_itfc, ctx);
    EXPECT_EQ(r_reward, -1.0);

    // actions from r state
    shared_ptr<ActionVector> r_actions_ptr_itfc = env.get_valid_actions_itfc(r_state_itfc, ctx);
    shared_ptr<StringActionVector> r_actions_ptr = convert_action_vector(r_actions_ptr_itfc);
    StringActionVector& r_actions = *r_actions_ptr;
    EXPECT_EQ(r_actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(r_actions, "left"));
    EXPECT_TRUE(action_vector_contains(r_actions, "down"));

    // r state is not sink
    EXPECT_FALSE(env.is_sink_state_itfc(r_state_itfc,ctx));

    // Move down
    shared_ptr<const State> rd_state_itfc = env.sample_transition_distribution_itfc(
        r_state_itfc, d_act_itfc, *manager_ptr, ctx);
    shared_ptr<const IntPairState> rd_state = static_pointer_cast<const IntPairState>(rd_state_itfc);
    EXPECT_EQ(rd_state->state.first, 1);
    EXPECT_EQ(rd_state->state.second, 1);

    // Check reward (cost) = -1
    double rd_reward = env.get_reward(r_state, d_act, ctx);
    EXPECT_EQ(rd_reward, -1.0);

    // no actions from rd_state as sink
    shared_ptr<ActionVector> rd_actions_ptr_itfc = env.get_valid_actions_itfc(rd_state_itfc, ctx);
    shared_ptr<StringActionVector> rd_actions_ptr = convert_action_vector(rd_actions_ptr_itfc);
    StringActionVector& rd_actions = *rd_actions_ptr;
    EXPECT_EQ(rd_actions.size(), 0ul);

    // rd state IS sink
    EXPECT_TRUE(env.is_sink_state_itfc(rd_state_itfc,ctx));


    // Also try move down from init state
    shared_ptr<const State> d_state_itfc = env.sample_transition_distribution_itfc(
        init_state_itfc, d_act_itfc, *manager_ptr, ctx);
    shared_ptr<const IntPairState> d_state = static_pointer_cast<const IntPairState>(d_state_itfc);
    EXPECT_EQ(d_state->state.first, 0);
    EXPECT_EQ(d_state->state.second, 1);

    // Check reward (cost) = -1
    double d_reward = env.get_reward_itfc(init_state_itfc, d_act_itfc, ctx);
    EXPECT_EQ(d_reward, -1.0);

    // actions from d state
    shared_ptr<ActionVector> d_actions_ptr_itfc = env.get_valid_actions_itfc(d_state_itfc, ctx);
    shared_ptr<StringActionVector> d_actions_ptr = convert_action_vector(d_actions_ptr_itfc);
    StringActionVector& d_actions = *d_actions_ptr;
    EXPECT_EQ(d_actions.size(), 2ul);
    EXPECT_TRUE(action_vector_contains(d_actions, "right"));
    EXPECT_TRUE(action_vector_contains(d_actions, "up"));

    // r state is not sink
    EXPECT_FALSE(env.is_sink_state(d_state,ctx));
}

/**
 * Currently not working and dont know why. Tried printing things out in the equals and hash functions of int pair 
 * state, and everything seems in order. Also they all pass when using the _itfc versions, which is what the 
 * thts routine will use... so leaving until later.
 */
TEST(Env_MdpImplementation, todo__test_get_transition_distribution__todo_fix_thts_type_subclasses_being_used_in_dicts)
{
    shared_ptr<const StringAction> l_act = make_shared<const StringAction>("left");
    shared_ptr<const StringAction> r_act = make_shared<const StringAction>("right");
    shared_ptr<const StringAction> d_act = make_shared<const StringAction>("down");
    shared_ptr<const StringAction> u_act = make_shared<const StringAction>("up");

    shared_ptr<const Action> l_act_itfc = static_pointer_cast<const Action>(l_act);
    shared_ptr<const Action> r_act_itfc = static_pointer_cast<const Action>(r_act);
    shared_ptr<const Action> d_act_itfc = static_pointer_cast<const Action>(d_act);
    shared_ptr<const Action> u_act_itfc = static_pointer_cast<const Action>(u_act);

    shared_ptr<const IntPairState> init_state = make_shared<IntPairState>(0,0);
    shared_ptr<const IntPairState> r_state = make_shared<IntPairState>(1,0);

    shared_ptr<const State> init_state_itfc = static_pointer_cast<const State>(init_state);
    shared_ptr<const State> r_state_itfc = static_pointer_cast<const State>(r_state);

    // Setup mock manager, testing fully observable env, so expect no calls to it
    shared_ptr<MockThtsManager> mock_manager_ptr = make_shared<MockThtsManager>();
    MockThtsManager& mock_manager = *mock_manager_ptr;
    shared_ptr<ThtsManager> manager_ptr = static_pointer_cast<ThtsManager>(mock_manager_ptr);
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(0);
    
    // Setup envs
    TestThtsEnv deter_env(1,0.0);
    TestThtsEnv stoch_env(1,0.25);

    // deterministic get transition distr
    ThtsEnvContext ctx;
    shared_ptr<IntPairStateDistr> deter_distr_ptr = deter_env.get_transition_distribution(init_state, r_act, ctx);
    IntPairStateDistr& deter_distr = *deter_distr_ptr;
    EXPECT_EQ(deter_distr[r_state], 1.0); // this doesnt work

    // stochastic get transition distr
    shared_ptr<IntPairStateDistr> stoch_distr_ptr = stoch_env.get_transition_distribution(init_state, r_act, ctx);
    IntPairStateDistr& stoch_distr = *stoch_distr_ptr;
    EXPECT_EQ(stoch_distr[r_state], 0.75); // this doesnt
    EXPECT_EQ(stoch_distr[init_state], 0.25); // this works

    // deterministic get transition distr, using the thts interface
    shared_ptr<StateDistr> deter_distr_ptr_itfc = deter_env.get_transition_distribution_itfc(
        init_state_itfc, r_act_itfc, ctx);
    StateDistr& deter_distr_itfc = *deter_distr_ptr_itfc;
    EXPECT_EQ(deter_distr_itfc[r_state_itfc], 1.0);

    // stochastic get transition distr
    shared_ptr<StateDistr> stoch_distr_ptr_itfc = stoch_env.get_transition_distribution_itfc(
        init_state_itfc, r_act_itfc, ctx);
    StateDistr& stoch_distr_itfc = *stoch_distr_ptr_itfc;
    EXPECT_EQ(stoch_distr_itfc[r_state_itfc], 0.75);
    EXPECT_EQ(stoch_distr_itfc[init_state_itfc], 0.25);
}



/**
 * Want to test the pomdp part of implementation, but 
 */
TEST(Env_PomdpImplementation, todo) 
{
    FAIL();
}