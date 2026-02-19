#include "helper.h"

#include "thts_context.h"
#include "thts_env.h"
#include "thts_manager.h"

using namespace std;

namespace thts {
    /**
     * Implementation of const heuristic function
     */
    ConstHeuristicFn::ConstHeuristicFn(double value) : value(value)
    {
    }

    double ConstHeuristicFn::operator()(
        shared_ptr<const State> state, ThtsEnv& env, ThtsManager& manager, int depth) 
    {
        return this->value;
    }

    /**
     * Implementation of the default zero heuristic function.
     */
    ZeroHeuristicFn::ZeroHeuristicFn() : ConstHeuristicFn(0.0)
    {
    }

    /**
     * Implementation of constant one heuristic function.
     */
    OneHeuristicFn::OneHeuristicFn() : ConstHeuristicFn(1.0)
    {
    }

    /**
     * Implementation of the rollout heuristic function.
     * TODO: move adding env + manager to constructor when working on v1.0
     */
    RolloutHeuristicFn::RolloutHeuristicFn()
    {
    }

    double RolloutHeuristicFn::operator()(
        shared_ptr<const State> state, ThtsEnv& env, ThtsManager& manager, int depth) 
    {
        ThtsContext& ctx = *manager.get_thts_context();
        int rollout_steps_left = manager.max_depth - depth;
        double rollout_reward = 0.0;

        while (rollout_steps_left-- > 0 && !env.is_sink_state_itfc(state, ctx)) {
            shared_ptr<ActionVector> actions = env.get_valid_actions_itfc(state, ctx);
            int index = manager.get_rand_int(0, actions->size());
            shared_ptr<const Action> action = actions->at(index);
            rollout_reward += env.get_reward_itfc(state, action, ctx);
            state = env.sample_transition_distribution_itfc(state, action, manager, ctx);
        }
        
        return rollout_reward;
    }
}

namespace thts::helper {
    std::shared_ptr<HeuristicFn> rollout_heuristic_fn = std::make_shared<RolloutHeuristicFn>();
    std::shared_ptr<HeuristicFn> zero_heuristic_fn = std::make_shared<ZeroHeuristicFn>();
    std::shared_ptr<HeuristicFn> one_heuristic_fn = std::make_shared<OneHeuristicFn>();

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