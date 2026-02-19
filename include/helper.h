#pragma once

#include "thts_types.h"

#include <memory>

// forward declares
namespace thts {
    class ThtsEnv;
    class ThtsManager;
}

namespace thts {

    /**
     * Abstract base class for heuristic functions
     */
    struct HeuristicFn 
    {
        public:
            virtual double operator()(
                std::shared_ptr<const State> state, 
                ThtsEnv& env, 
                ThtsManager& manager, 
                int depth=0) = 0;

            virtual ~HeuristicFn() = default;
    };

    /**
     * A heuristic function that returns a constant value
     */
    struct ConstHeuristicFn : public HeuristicFn
    {
        public:
            ConstHeuristicFn(double value);
            virtual double operator()(
                std::shared_ptr<const State> state, 
                ThtsEnv& env, 
                ThtsManager& manager, 
                int depth=0) override;
            virtual ~ConstHeuristicFn() = default;

        private:
            double value;
    };

    /**
     * A default heuristic function that returns a constant zero
     */
    struct ZeroHeuristicFn : public ConstHeuristicFn
    {
        public:
            ZeroHeuristicFn();
            virtual ~ZeroHeuristicFn() = default;
    };

    /**
     * A default heuristic function that returns a constant one
     */
    struct OneHeuristicFn : public ConstHeuristicFn
    {
        public:
            OneHeuristicFn();
            virtual ~OneHeuristicFn() = default;
    };
    
    /**
     * The rollout heuristic function, that returns an MC estimate of 'state' with a rollout with random policy
     */
    struct RolloutHeuristicFn : public HeuristicFn
    {
        public:
            RolloutHeuristicFn();
            virtual double operator()(
                std::shared_ptr<const State> state, 
                ThtsEnv& env, 
                ThtsManager& manager, 
                int depth=0) override;
            virtual ~RolloutHeuristicFn() = default;
    };
}

namespace thts::helper {
    /**
     * String split function
     */
    std::vector<std::string> string_split(const std::string& s, const std::string& delimiter=",");

    /** Shared instances for use as default heuristic (e.g. in run_manager). */
    extern std::shared_ptr<HeuristicFn> rollout_heuristic_fn;
    extern std::shared_ptr<HeuristicFn> zero_heuristic_fn;
    extern std::shared_ptr<HeuristicFn> one_heuristic_fn;
}