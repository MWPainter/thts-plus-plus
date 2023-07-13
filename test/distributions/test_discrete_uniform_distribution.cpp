#include "test/distributions/test_discrete_uniform_distribution.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "distributions/discrete_uniform_distribution.h"

// includes
#include "test/test_thts_manager.h"

#include <memory>
#include <vector>


using namespace std;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for assertations)
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::StrEq;

/**
 * Testing discrete uniform distrubtion
 */
TEST(DistributionsDiscreteUniform_UnitTest, reminder_to_do) {
    FAIL();
}


TEST(DistributionsDiscreteUniform_Sampling, test_sample) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_int(0,4))
        .Times(6)
        .WillOnce(Return(0))
        .WillOnce(Return(1))
        .WillOnce(Return(0))
        .WillOnce(Return(3))
        .WillOnce(Return(2))
        .WillOnce(Return(2));

    shared_ptr<vector<int>> keys = make_shared<vector<int>>();
    keys->push_back(10);
    keys->push_back(11);
    keys->push_back(12);
    keys->push_back(13);
    DiscreteUniformDistribution distr(keys);

    EXPECT_EQ(distr.sample(mock_manager), 10);
    EXPECT_EQ(distr.sample(mock_manager), 11);
    EXPECT_EQ(distr.sample(mock_manager), 10);
    EXPECT_EQ(distr.sample(mock_manager), 13);
    EXPECT_EQ(distr.sample(mock_manager), 12);
    EXPECT_EQ(distr.sample(mock_manager), 12);
}