#include "test_thts_env_context.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// testing
#include "thts_env_context.h"

// includes
#include <string>


using namespace std;
using namespace thts;

/**
 * Testing herlper::get_max_key_break_ties_randomly
 * From helper_templates.h
 */
TEST(EnvContext_CheckIsMapWithoutDefaults, test_normal_use) 
{
    ThtsEnvContext context;
    context.put_value("Hello,", make_shared<double>(1.0));
    context.put_value("World!", make_shared<double>(2.5));
    EXPECT_EQ(context.get_value<double>("Hello,"), 1.0);
    EXPECT_EQ(context.get_value<double>("World!"), 2.5);
}

TEST(EnvContext_CheckIsMapWithoutDefaults, test_missing_key) 
{
    ThtsEnvContext context;
    context.put_value("Hello,", make_shared<double>(1.0));
    context.put_value("World!", make_shared<double>(2.5));
    EXPECT_ANY_THROW(context.get_value_raw("NotInContext"));
}
