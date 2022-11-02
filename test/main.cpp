#include "test_thts_env.h"
#include "test_thts_nodes.h"

#include <iostream>

using namespace std;
using namespace thts_test;

/**
 * Main function collates and runs all tests in the test directory
 */
int main(int argc, char *argv[]) {
    run_thts_env_tests();
    cout << endl << endl << endl;
    run_thts_node_tests();
    return 0;
}

