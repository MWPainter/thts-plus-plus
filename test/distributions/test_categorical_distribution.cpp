#include "test/distributions/test_categorical_distribution.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "distributions/categorical_distribution.h"

// includes
#include "thts_manager.h"

#include <memory>
#include <unordered_map>


using namespace std;
using namespace thts;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for assertations)
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::StrEq;



TEST(DistributionsCategorical_UnitTest, reminder_to_do) {
    FAIL();
}





TEST(DistributionsCategorical_Sampling, test_sample_binomial_naive) 
{   
    RandManager rand_manager;
    shared_ptr<unordered_map<int,double>> distr_map = make_shared<unordered_map<int,double>>();
    distr_map->insert_or_assign(0,0.3);
    distr_map->insert_or_assign(1,0.7);

    CategoricalDistribution distr(distr_map);

    // sample
    int num_zeros = 0;
    for (int i=0; i<10000; i++) {
        if (distr.sample(rand_manager) == 0) {
            num_zeros++;
        }
    }

    // less than 1e-7 chance of failing
    EXPECT_LT(num_zeros, 3250);
    EXPECT_GT(num_zeros, 2750);

    // update distr
    distr.update(0, 20.0);
    distr.update(1, 80.0);

    // sample
    num_zeros = 0;
    for (int i=0; i<10000; i++) {
        if (distr.sample(rand_manager) == 0) {
            num_zeros++;
        }
    }

    // less than 1e-7 chance of failing
    EXPECT_LT(num_zeros, 2250);
    EXPECT_GT(num_zeros, 1750);
}

TEST(DistributionsCategorical_Sampling, test_sample_binomial_with_alias) 
{   
    RandManager rand_manager;
    shared_ptr<unordered_map<int,double>> distr_map = make_shared<unordered_map<int,double>>();
    distr_map->insert_or_assign(0,0.3);
    distr_map->insert_or_assign(1,0.7);

    CategoricalDistribution distr(distr_map, true, 2);

    // sample
    int num_zeros = 0;
    for (int i=0; i<10000; i++) {
        if (distr.sample(rand_manager) == 0) {
            num_zeros++;
        }
    }

    // less than 1e-7 chance of failing
    EXPECT_LT(num_zeros, 3250); 
    EXPECT_GT(num_zeros, 2750);

    // update one item (shouldn't change the distribution)
    distr.update(0, 20.0);

    // sample
    num_zeros = 0;
    for (int i=0; i<10000; i++) {
        if (distr.sample(rand_manager) == 0) {
            num_zeros++;
        }
    }

    // less than 1e-7 chance of failing
    // alias table should still have weights 0.3, 0.7, even though a weight was updated to 20.0
    EXPECT_LT(num_zeros, 3250);  
    EXPECT_GT(num_zeros, 2750); 

    // update second item, should now change
    distr.update(1, 80.0);

    // sample
    num_zeros = 0;
    for (int i=0; i<10000; i++) {
        if (distr.sample(rand_manager) == 0) {
            num_zeros++;
        }
    }

    // less than 1e-7 chance of failing
    EXPECT_LT(num_zeros, 2250);
    EXPECT_GT(num_zeros, 1750);
}

TEST(DistributionsCategorical_Eyeball_Sampling, test_sample_harder_naive) 
{   
    RandManager rand_manager;
    shared_ptr<unordered_map<int,double>> distr_map = make_shared<unordered_map<int,double>>();
    distr_map->insert_or_assign(0,0.05);
    distr_map->insert_or_assign(1,0.3);
    distr_map->insert_or_assign(2,0.2);
    distr_map->insert_or_assign(3,0.3);
    distr_map->insert_or_assign(4,0.15);

    CategoricalDistribution distr(distr_map);

    // sample
    unordered_map<int,double> counts;
    for (int i=0; i<10000; i++) {
        counts[distr.sample(rand_manager)]++;
    }

    cout << "Expect counts approx: [500,3000,2000,3000,1500]:" << endl;
    cout << "Actual count are: [" 
        << counts[0] << "," 
        << counts[1] << "," 
        << counts[2] << "," 
        << counts[3] << ","
        << counts[4] << "]" << endl;
}

TEST(DistributionsCategorical_Eyeball_Sampling, test_sample_harder_with_alias) 
{   
    RandManager rand_manager;
    shared_ptr<unordered_map<int,double>> distr_map = make_shared<unordered_map<int,double>>();
    distr_map->insert_or_assign(0,0.05);
    distr_map->insert_or_assign(1,0.3);
    distr_map->insert_or_assign(2,0.2);
    distr_map->insert_or_assign(3,0.3);
    distr_map->insert_or_assign(4,0.15);

    CategoricalDistribution distr(distr_map, true);

    // sample
    unordered_map<int,double> counts;
    for (int i=0; i<10000; i++) {
        counts[distr.sample(rand_manager)]++;
    }

    cout << "Expect counts approx: [500,3000,2000,3000,1500]:" << endl;
    cout << "Actual count are: [" 
        << counts[0] << "," 
        << counts[1] << "," 
        << counts[2] << "," 
        << counts[3] << ","
        << counts[4] << "]" << endl;
}