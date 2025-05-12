#include "helper.h"

using namespace std;

namespace thts::helper {
    /**
     * Implementation of the default zero heuristic function.
     */
    double zero_heuristic_fn(shared_ptr<const State> state, shared_ptr<ThtsEnv> env) {
        return 0.0;
    }

    /**
     * String split function, adapted from stack overflow comment:
     * https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
     */
    vector<string> string_split(const std::string& s, const std::string& delimiter)
    {
        vector<string> result;
        size_t last = 0; 
        size_t next = s.find(delimiter, last); 
        while (next != string::npos) 
        {   
            result.push_back(s.substr(last, next-last));
            last = next + 1; 
            next = s.find(delimiter, last);
        }
        result.push_back(s.substr(last));
        return result;
    }
}