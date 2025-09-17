#pragma once

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thts {
    // forward declare
    class ThtsPool;


    /**
     * v1TODO: want to move as much shared logic into here as we can
     * 
     * Base class for all nodes.
     * 
     * Member variables:
     *      lock: 
     *          A mutex that is used to protect this entire node.
     */
    class ThtsNode : public std::enable_shared_from_this<ThtsNode> {
        // Allow ThtsCNode, Logger and Pool access to private members
        friend ThtsPool;

        protected:
            std::mutex lock;
        
        public:

            /**
             * Constructor
             */
            ThtsNode();

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~ThtsNode() = default;
    };
}