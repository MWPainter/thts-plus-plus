#pragma once

#include "thts_env.h"
#include "thts_types.h"

#include "go/go_state_action.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "KataGo/cpp/core/config_parser.h"
#include "KataGo/cpp/core/logger.h"
#include "KataGo/cpp/neuralnet/nneval.h"
#include "KataGo/cpp/neuralnet/nninputs.h"

namespace thts {
    // Location of the neural net file
    const std::string NN_FILENAME = "external/kata1-b40c256-s11840935168-d2898845681.bin.gz";

    // Typedefs for long types
    typedef std::vector<std::shared_ptr<const GoAction>> GoActionVector;
    typedef std::unordered_map<std::shared_ptr<const GoAction>,double> GoActionPolicy;
    typedef std::unordered_map<std::shared_ptr<const GoState>,double> GoStateDistr;


    /** 
     * An interface for Thts to be able to interact with the KataGo implementation.
     * 
     * Because multi-stone suicide is usually not allowed, we set the rules to disallow multi-stone suicide.
     * 
     * Member variables:
     *      board_size: The size of the go board to use
     *      komi: The komi to use (komi = go term)
     *      init_board: The initial go board (empty)
     *      rules: A KataGo rules object specifying the rules to be played with
     *      init_state: A cached init state 
     *      nn_eval: KataGo's neural net evaluator object
     *      _cfg: Required in nn_eval construction, just use/pass empty
     *      _logger: Required in nn_eval construction, just use/pass empty
     *      nn_temp: A temperature to use when getting probabilities from the nn policy
     *      reward_scale: A scaling to use for rewards, the outputs from nn value function is in range [-1,1]
     *      cur_board_size: Globally accessable board size to get around annoying passing of values to action pretty 
     *          print function
     */
    class GoEnv : public ThtsEnv {

        protected:
            int board_size;
            float komi;
            Board init_board;
            Rules rules;
            std::shared_ptr<GoState> init_state;

            NNEvaluator* nn_eval;
            ConfigParser _cfg;
            Logger _logger;

            float nn_temp;
            double reward_scale;
            double dynamic_score_center;

        public:
            static int cur_board_size;

        /**
         * Functions to imlpement the ThtsEnv interface.
         */
        public:
            GoEnv(
                int board_size, 
                float komi, 
                bool use_nn_eval=true, 
                std::string nn_eval_rand_seed="60415", 
                float nn_temp=1.0,
                double reward_scale=100.0,
                double dynamic_score_center=0.0);
            virtual ~GoEnv();

            NNEvaluator* get_nn_eval();
            Logger* get_logger();

            std::shared_ptr<const GoState> get_initial_state() const;

            bool is_sink_state(std::shared_ptr<const GoState> state) const;

            std::shared_ptr<GoActionVector> get_valid_actions(std::shared_ptr<const GoState> state) const;

            std::shared_ptr<GoStateDistr> get_transition_distribution(
                std::shared_ptr<const GoState> state, std::shared_ptr<const GoAction> action) const;

            std::shared_ptr<const GoState> sample_transition_distribution(
                std::shared_ptr<const GoState> state, std::shared_ptr<const GoAction> action) const;

            double get_reward(
                std::shared_ptr<const GoState> state, 
                std::shared_ptr<const GoAction> action, 
                std::shared_ptr<const GoState> observation) const;
        
        /**
         * Neural Net interface
         */
        private:
            void fill_cached_values_with_nn_output(std::shared_ptr<const GoState> state);
        
        public:
            void update_dynamic_score_center_for_root_state(std::shared_ptr<const GoState> state);
            double get_heuristic_val_from_nn(std::shared_ptr<const GoState> state);
            double get_heuristic_val_from_nn_itfc(std::shared_ptr<const State> state);
            std::shared_ptr<GoActionPolicy> get_policy_from_nn(std::shared_ptr<const GoState> state);
            std::shared_ptr<ActionPrior> get_policy_from_nn_itfc(std::shared_ptr<const State> state);
            double get_black_win_prob_from_nn(std::shared_ptr<const GoState> state);

        /**
         * Boilerplate functinos (defined in thts_env_template.{h,cpp}) using the default implementations provided by 
         * thts_env.{h,cpp}. 
         */
        public:
            virtual std::shared_ptr<GoStateDistr> get_observation_distribution(
                std::shared_ptr<const GoAction> action, std::shared_ptr<const GoState> next_state) const;
            virtual std::shared_ptr<const GoState> sample_observation_distribution(
                std::shared_ptr<const GoAction> action, 
                std::shared_ptr<const GoState> next_state, 
                RandManager& rand_manager) const;
            virtual std::shared_ptr<ThtsEnvContext> sample_context(std::shared_ptr<const GoState> state) const;

        /**
         * The ThtsEnv interface.
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                RandManager& rand_manager) const;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const;
            virtual std::shared_ptr<ThtsEnvContext> sample_context_itfc(std::shared_ptr<const State> state) const;
    };

    /**
     * Heuristic fn 
    */
    double go_heuristic_fn(std::shared_ptr<const State> state, std::shared_ptr<ThtsEnv> env);

    /**
     * Prior fn
    */
    std::shared_ptr<ActionPrior> go_prior_fn(std::shared_ptr<const State> state, std::shared_ptr<ThtsEnv> env);
}
