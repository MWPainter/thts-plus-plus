#include "test_helpers.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "helper_templates.h"

// includes
#include "test_thts_manager.h"

#include <string>
#include <unordered_map>


using namespace std;
using namespace thts;
using namespace thts_test;

// actions (for 'EXPECT_CALL')
using ::testing::Return;

// matchers (for assertations)
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::StrEq;

/**
 * Testing herlper::get_max_key_break_ties_randomly
 * From helper_templates.h
 */
TEST(Helpers_GetMaxKeyBreakTiesRandomly, test_integers_unique_max) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);

    unordered_map<int,int> map;
    map[0] = -100;
    map[1] = 12;
    map[2] = 200;

    EXPECT_EQ(helper::get_max_key_break_ties_randomly(map,mock_manager), 2);

    map[1] = 60415;
    map[2] = 6041;
    map[0] = 604;

    EXPECT_EQ(helper::get_max_key_break_ties_randomly(map,mock_manager), 1);
}

TEST(Helpers_GetMaxKeyBreakTiesRandomly, test_doubles_unique_max) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_int)
        .Times(0);

    unordered_map<int,int> map;
    map[0] = -100;
    map[1] = 12;
    map[2] = 200;

    EXPECT_EQ(helper::get_max_key_break_ties_randomly(map,mock_manager), 2);

    map[0] = 604;
    map[1] = 6041;
    map[2] = 60415;
    map[3] = 6041;
    map[4] = 604;

    EXPECT_EQ(helper::get_max_key_break_ties_randomly(map,mock_manager), 2);
}

TEST(Helpers_GetMaxKeyBreakTiesRandomly, test_integers_tied_max) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_int(0,2))
        .Times(4)
        .WillOnce(Return(0))
        .WillOnce(Return(1))
        .WillOnce(Return(0))
        .WillOnce(Return(1));

    unordered_map<int,int> map;
    map[0] = 12;
    map[1] = -10;
    map[2] = 12;
    map[3] = 1;

    vector<int> results;
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    EXPECT_THAT(results, Contains(0));
    EXPECT_THAT(results, Contains(2));

    map[0] = 604;
    map[1] = 6041;
    map[2] = 60415;
    map[3] = 60415;
    map[4] = 6041;

    results[0] = helper::get_max_key_break_ties_randomly(map,mock_manager);
    results[1] = helper::get_max_key_break_ties_randomly(map,mock_manager);
    EXPECT_THAT(results, Contains(2));
    EXPECT_THAT(results, Contains(3));
}

TEST(Helpers_GetMaxKeyBreakTiesRandomly, test_doubles_tied_max) 
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_int(0,3))
        .Times(3)
        .WillOnce(Return(0))
        .WillOnce(Return(1))
        .WillOnce(Return(2));
    EXPECT_CALL(mock_manager, get_rand_int(0,2))
        .Times(2)
        .WillOnce(Return(0))
        .WillOnce(Return(1));

    unordered_map<int,double> map;
    map[0] = 20.0;
    map[1] = 12.0;
    map[2] = 20.0;
    map[3] = -1.0;
    map[4] = 20.0;

    vector<int> results;
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    EXPECT_THAT(results, Contains(0));
    EXPECT_THAT(results, Contains(2));
    EXPECT_THAT(results, Contains(4));

    map[0] = 6.04;
    map[1] = 60.41;
    map[2] = 6.0415;
    map[3] = 60.41;
    map[4] = 6.04;

    results = vector<int>();
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    results.push_back(helper::get_max_key_break_ties_randomly(map,mock_manager));
    EXPECT_THAT(results, Contains(1));
    EXPECT_THAT(results, Contains(3));
}



/**
 * Testing herlper::sample_from_distribution
 * From helper_templates.h
 */
shared_ptr<unordered_map<int,int>> call_sampling(
    unordered_map<int,double>& distr, ThtsManager& manager, bool normalised) 
{
    shared_ptr<unordered_map<int,int>> counts_ptr = make_shared<unordered_map<int,int>>();
    unordered_map<int,int>& counts = *counts_ptr;
    for (int i=0; i<10; i++) {
        int indx = helper::sample_from_distribution(distr, manager, normalised);
        counts[indx]++;
    }
    return counts_ptr;
}

TEST(Helpers_SampleFromDistribution, typical_sampling_normalised)
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(10)
        .WillOnce(Return(0.05))
        .WillOnce(Return(0.15))
        .WillOnce(Return(0.25))
        .WillOnce(Return(0.35))
        .WillOnce(Return(0.45))
        .WillOnce(Return(0.55))
        .WillOnce(Return(0.65))
        .WillOnce(Return(0.75))
        .WillOnce(Return(0.85))
        .WillOnce(Return(0.95));

    unordered_map<int,double> distr;
    distr[0] = 0.4;
    distr[1] = 0.1;
    distr[2] = 0.3;
    distr[3] = 0.2;
    shared_ptr<unordered_map<int,int>> counts = call_sampling(distr, mock_manager, true);

    EXPECT_EQ(counts->at(0), 4);    
    EXPECT_EQ(counts->at(1), 1);    
    EXPECT_EQ(counts->at(2), 3);    
    EXPECT_EQ(counts->at(3), 2);
}

TEST(Helpers_SampleFromDistribution, typical_sampling_unormalised)
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(10)
        .WillOnce(Return(0.05))
        .WillOnce(Return(0.15))
        .WillOnce(Return(0.25))
        .WillOnce(Return(0.35))
        .WillOnce(Return(0.45))
        .WillOnce(Return(0.55))
        .WillOnce(Return(0.65))
        .WillOnce(Return(0.75))
        .WillOnce(Return(0.85))
        .WillOnce(Return(0.95));

    unordered_map<int,double> distr;
    distr[0] = 4.0;
    distr[1] = 1.0;
    distr[2] = 3.0;
    distr[3] = 2.0;
    shared_ptr<unordered_map<int,int>> counts = call_sampling(distr, mock_manager, false);

    EXPECT_EQ(counts->at(0), 4);    
    EXPECT_EQ(counts->at(1), 1);    
    EXPECT_EQ(counts->at(2), 3);    
    EXPECT_EQ(counts->at(3), 2);
}

TEST(Helpers_SampleFromDistribution, error_check_too_much_mass)
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(2)
        .WillOnce(Return(0.15))
        .WillOnce(Return(0.75));

    unordered_map<int,double> distr;
    distr[0] = 4.0;
    distr[1] = 1.0;
    distr[2] = 3.0;

    EXPECT_ANY_THROW(call_sampling(distr, mock_manager, true));

    distr[0] = 0.5;
    distr[1] = 0.5;
    distr[2] = 0.5;

    EXPECT_ANY_THROW(call_sampling(distr, mock_manager, true));
}

TEST(Helpers_SampleFromDistribution, error_check_too_little_mass)
{
    MockThtsManager mock_manager;
    EXPECT_CALL(mock_manager, get_rand_uniform)
        .Times(1)
        .WillOnce(Return(0.95));

    unordered_map<int,double> distr;
    distr[0] = 0.04;
    distr[1] = 0.01;
    distr[2] = 0.03;

    EXPECT_ANY_THROW(call_sampling(distr, mock_manager, true));
}



/**
 * Testing herlper::vector_pretty_print_string
 * From helper_templates.h
 */
TEST(Helpers_VectorPrettyPrint, check_against_static) 
{
    vector<string> vec;
    vec.push_back("Hello");
    vec.push_back("World!");
    EXPECT_THAT(helper::vector_pretty_print_string(vec), StrEq("[Hello,World!]"));
}



/**
 * Testing herlper::unordered_map_pretty_print_string
 * From helper_templates.h
 */
TEST(Helpers_UnorderedMapPrettyPrint, check_against_static) 
{
    unordered_map<string,string> map;
    map["Hel"] = "lo";
    map["Wor"] = "ld!";
    EXPECT_THAT(helper::unordered_map_pretty_print_string(map), 
        AnyOf(StrEq("{Hel:lo,Wor:ld!}"),StrEq("{Wor:ld!,Hel:lo}")));

}


