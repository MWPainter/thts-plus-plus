#include "test/distributions/test_mixed_distribution.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "distributions/mixed_distribution.h"

// includes
#include "distributions/discrete_uniform_distribution.h"
#include "test/test_thts_manager.h"

#include <memory>
#include <vector>
#include <unordered_map>


using namespace std;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for assertations)
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::StrEq;



TEST(DistributionsMixed_UnitTest, reminder_to_do) {
    FAIL();
}





TEST(DistributionsMixed_Sampling, test_sample) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_uniform())
        .Times(6)
        .WillOnce(Return(0.1))
        .WillOnce(Return(0.45))
        .WillOnce(Return(0.9))
        .WillOnce(Return(0.2))
        .WillOnce(Return(0.55))
        .WillOnce(Return(0.95));
    EXPECT_CALL(mock_manager, get_rand_int(0,1))
        .Times(6)
        .WillRepeatedly(Return(0));

    shared_ptr<vector<int>> keys1 = make_shared<vector<int>>();
    keys1->push_back(0);
    shared_ptr<vector<int>> keys2 = make_shared<vector<int>>();
    keys2->push_back(1);
    shared_ptr<vector<int>> keys3 = make_shared<vector<int>>();
    keys3->push_back(2);

    shared_ptr<DiscreteUniformDistribution<int>> distr1 = make_shared<DiscreteUniformDistribution<int>>(keys1);
    shared_ptr<DiscreteUniformDistribution<int>> distr2 = make_shared<DiscreteUniformDistribution<int>>(keys2);
    shared_ptr<DiscreteUniformDistribution<int>> distr3 = make_shared<DiscreteUniformDistribution<int>>(keys3);

    shared_ptr<MixedDistributionDistr<int>> mix_distr_map = make_shared<MixedDistributionDistr<int>>();
    mix_distr_map->insert_or_assign(distr1, 0.3);
    mix_distr_map->insert_or_assign(distr2, 0.4);
    mix_distr_map->insert_or_assign(distr3, 0.3);

    MixedDistribution mix_distr(mix_distr_map);
    
    unordered_map<int, int> counts;
    for (int i=0; i<6; i++) {
        counts[mix_distr.sample(mock_manager)]++;
    }
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 2);
}