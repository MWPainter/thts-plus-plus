#include "test/algorithms/test_max_heap.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "algorithms/common/max_heap.h"

// includes
#include <unordered_map>


using namespace std;
using namespace thts;
using namespace thts::test;

// actions (for 'EXPECT_CALL')
using ::testing::Contains;


/**
 * Test that heap can be used to implement heapsort pretty much
 */
TEST(MaxHeap_IntegrationTests, heapsort) {
    MockMaxHeap<int> maxheap(10);
    unordered_map<int,double> init_heap = {
        {0, 0.5},
        {1, 1.5},
        {2, 2.5},
        {3, 3.5},
        {7, 7.5},
        {8, 8.5},
        {9, 9.5},
        {4, 4.5},
        {5, 5.5},
        {6, 6.5},
    };
    maxheap.fill_and_heapify(init_heap);

    for (int i=9; i>=0; i--) {
        int key = maxheap.peek_top_key();
        double val = maxheap.peek_top_value();
        ASSERT_EQ(i, key);
        ASSERT_EQ(i+0.5, val);
        maxheap.pop();
    }
}


/**
 * Test that heap can be used to implement heapsort pretty much
 */
TEST(MaxHeap_IntegrationTests, heapsort_slow_construction) {
    MockMaxHeap<int> maxheap(10);
    unordered_map<int,double> init_heap = {
        {0, 0.5},
        {1, 1.5},
        {2, 2.5},
        {3, 3.5},
        {7, 7.5},
        {8, 8.5},
        {9, 9.5},
        {4, 4.5},
        {5, 5.5},
        {6, 6.5},
    };
    for (pair<int,double> pr : init_heap) {
        maxheap.insert_or_assign(pr.first, pr.second);
    }

    for (int i=9; i>=0; i--) {
        int key = maxheap.peek_top_key();
        double val = maxheap.peek_top_value();
        ASSERT_EQ(i, key);
        ASSERT_EQ(i+0.5, val);
        maxheap.pop();
    }
}

/**
 * Test sifting (up and down)
 */
TEST(MaxHeap_IntegrationTests, sifting_up_and_down) {
    MockMaxHeap<int> maxheap(10);
    unordered_map<int,double> init_heap = {
        {40, 40.0},
        {25, 25.0},
        {22, 22.0},
        {0, 0.0},
        {1, 1.0},
        {2, 2.0},
    };
    maxheap.fill_and_heapify(init_heap);

    maxheap.insert_or_assign(0, 30.0);   
    vector<int> keys = { maxheap.peek_key(1), maxheap.peek_key(2) };
    EXPECT_THAT(keys, Contains(0));

    maxheap.insert_or_assign(0, -30.0);   
    keys = { maxheap.peek_key(3), maxheap.peek_key(4), maxheap.peek_key(5) };
    EXPECT_THAT(keys, Contains(0));
}