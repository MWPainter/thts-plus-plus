#include "algorithms/common/ent_chance_node.h"

#include "helper_templates.h"

#include <memory>
#include <unordered_map>

using namespace std;

namespace thts {

    /**
     * Entropy = expected value of child entropies (i.e. empirical average)
     * 
     * Adapted from DPDNode DPBackup function
    */
    void EntCNode::backup_ent_impl(EntDNodeChildMap& children, EmpiricalDistributionMap& empirical_distribution) {
        num_backups++;

        subtree_entropy = 0.0;
        double sum_child_n_selections = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<EntDNode>> pr : children) {
            shared_ptr<const Observation> observation = pr.first;
            EntDNode& child = (EntDNode&) *pr.second;
            double child_n_selections = empirical_distribution[observation];
            if (child_n_selections == 0) continue;
            sum_child_n_selections += child_n_selections;
            subtree_entropy *= (sum_child_n_selections - child_n_selections) / sum_child_n_selections;
            subtree_entropy += child_n_selections * child.subtree_entropy / sum_child_n_selections;
        }
    }
}