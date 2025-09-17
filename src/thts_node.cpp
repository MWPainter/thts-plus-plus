#include "thts_node.h"

using namespace std;
using namespace thts;


namespace thts {
    /**
     * Constructor mostly uses initialisation list. 
     * 
     * Nuance use of heuristic value is to enforce nodes for sink states to have a value of zero
     */
    ThtsNode::ThtsNode() :
            lock()
    {
    }
}