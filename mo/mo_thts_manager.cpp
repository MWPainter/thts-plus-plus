#include "mo/mo_thts_manager.h"

#include <sstream>

using namespace std;

namespace thts {


    /**
     * Constructor. Initialises values directly other than random number generation.
     */    
    MoThtsManager::MoThtsManager(const MoThtsManagerArgs& args) :
        ThtsManager(args),
        reward_dim(args.reward_dim),
        mo_heuristic_fn(args.mo_heuristic_fn),
        heuristic_psuedo_trials(args.heuristic_psuedo_trials),
        use_vector_visit_counts(args.use_vector_visit_counts),
        convex_hull_max_size(args.convex_hull_max_size),
        convex_hull_tolerance(args.convex_hull_tolerance),
        use_solved_labelling(args.use_solved_labelling),
        solved_labelling_fail_confidence(args.solved_labelling_fail_confidence),
        solved_labelling_tolerance(args.solved_labelling_tolerance)
    {
        MoThtsEnv& mo_thts_env = *dynamic_pointer_cast<MoThtsEnv>(thts_env()); 
        if (reward_dim == MoThtsManagerArgs::reward_dim_default) {
            reward_dim = mo_thts_env.get_reward_dim();
        } else if (reward_dim != mo_thts_env.get_reward_dim()) {
            throw std::runtime_error("Reward dim in MoThtsManager doesn't match reward dim of env.");
        }

        if (mo_heuristic_fn == nullptr) {
            mo_heuristic_fn = get_default_mo_zero_heuristic_fn();
        }
    }

    std::shared_ptr<MoHeuristicFn> MoThtsManager::get_default_mo_zero_heuristic_fn() {
        return make_shared<MoZeroHeuristicFn>(reward_dim);
    }
}