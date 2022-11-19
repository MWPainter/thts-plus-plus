#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "thts_manager.h"

namespace thts_test {
    /**
     * Mock ThtsManager. Used to spoof random number generation so fixed for testing.
     */
    class MockThtsManager : public thts::ThtsManager {
        public:
            MOCK_METHOD(int, get_rand_int, (int min_included, int max_excluded), (override));
            MOCK_METHOD(double, get_rand_uniform, (), (override));
    };
}

