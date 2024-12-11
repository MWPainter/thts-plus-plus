#include "benchmark.h"

#include <iostream>
#include <string>

using namespace std;
using namespace thts_test;

/**
 * Main function collates and runs all tests in the test/go directory
 */
int main(int argc, char *argv[]) {
    go_state_benchmark();
    return 0;
}