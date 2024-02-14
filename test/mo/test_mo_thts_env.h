#pragma once

#include "thts_types.h"
#include "mo/mo_thts_env.h"

namespace thts::test{
    using namespace std;
    using namespace thts;


    const int RIGHT = 0;
    const int DOWN = 1;


    /** 
     * A MoThtsEnv that will be used throughout testing. We use these tests to test this class and the interface.
     */
    class TestMoThtsEnv : public MoThtsEnv {

        private:
            int walk_len;
            double stay_prob;
            bool add_extra_rewards;
            double new_dir_bonus;
            double same_dir_bonus;
            double gamma;

        /**
         * Node implementation
         */
        public:
            TestMoThtsEnv(
                int walk_len, 
                double stay_prob=0.0, 
                bool add_extra_rewards=false, 
                double new_dir_bonus=0.5, 
                double same_dir_bonus=0.3,
                double gamma=0.5) : 
                    ThtsEnv(true),
                    MoThtsEnv(add_extra_rewards ? 4 : 2,true), 
                    walk_len(walk_len), 
                    stay_prob(stay_prob), 
                    add_extra_rewards(add_extra_rewards),
                    new_dir_bonus(new_dir_bonus), 
                    same_dir_bonus(same_dir_bonus),
                    gamma(gamma)
            {
            } 

            TestMoThtsEnv(TestMoThtsEnv& other) : 
                ThtsEnv(true),
                MoThtsEnv(other.add_extra_rewards ? 4 : 2,true), 
                walk_len(other.walk_len), 
                stay_prob(other.stay_prob), 
                add_extra_rewards(other.add_extra_rewards),
                new_dir_bonus(other.new_dir_bonus),
                same_dir_bonus(other.same_dir_bonus),
                gamma(other.gamma)
            {
            }

            virtual ~TestMoThtsEnv() = default;

            virtual std::shared_ptr<ThtsEnv> clone() override {
                return std::dynamic_pointer_cast<ThtsEnv>(std::make_shared<TestMoThtsEnv>(*this));
            }

            double get_gamma() const {
                return gamma;
            }

            shared_ptr<const Int3TupleState> get_initial_state() const {
                return make_shared<Int3TupleState>(Int3TupleState(0,0,-1));
            }

            bool is_sink_state(shared_ptr<const Int3TupleState> state, ThtsEnvContext& ctx) const {
                return (std::get<0>(state->state) + std::get<1>(state->state)) == walk_len;
            }

            shared_ptr<IntActionVector> get_valid_actions(
                shared_ptr<const Int3TupleState> state, ThtsEnvContext& ctx) const 
            {
                shared_ptr<IntActionVector> valid_actions = make_shared<IntActionVector>();
                if (is_sink_state(state,ctx)) {
                    return valid_actions;
                }
                valid_actions->push_back(make_shared<const IntAction>(RIGHT));
                valid_actions->push_back(make_shared<const IntAction>(DOWN));
                return valid_actions;
            }

        private:
            shared_ptr<const Int3TupleState> make_candidate_next_state(
                shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, bool stay) const
            {
                shared_ptr<Int3TupleState> new_state = make_shared<Int3TupleState>(state->state);
                if (!stay) {
                    if (action->action == RIGHT) {
                        std::get<0>(new_state->state) += 1;
                    } else if (action->action == DOWN) {
                        std::get<1>(new_state->state) += 1;
                    }
                }
                std::get<2>(new_state->state) = action->action;
                return new_state;
            }

        public:
            shared_ptr<Int3TupleStateDistr> get_transition_distribution(
                shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, ThtsEnvContext& ctx) const 
            {
                shared_ptr<const Int3TupleState> new_state = make_candidate_next_state(state, action, false);
                shared_ptr<Int3TupleStateDistr> transition_distribution = make_shared<Int3TupleStateDistr>(); 
                transition_distribution->insert_or_assign(new_state, 1.0-stay_prob);
                if (stay_prob > 0.0) {
                    shared_ptr<const Int3TupleState> stay_state = make_candidate_next_state(state, action, true);
                    transition_distribution->insert_or_assign(stay_state, stay_prob);
                }
                return transition_distribution;
            }

            shared_ptr<const Int3TupleState> sample_transition_distribution(
                shared_ptr<const Int3TupleState> state, 
                shared_ptr<const IntAction> action, 
                RandManager& rand_manager,
                ThtsEnvContext& ctx) const 
            {
                if (stay_prob > 0.0) {
                    double sample = rand_manager.get_rand_uniform();
                    if (sample < stay_prob) {
                        return make_candidate_next_state(state,action,true);
                    }
                }
                return make_candidate_next_state(state,action,false);
            }

            Eigen::ArrayXd get_mo_reward(
                shared_ptr<const Int3TupleState> state, 
                shared_ptr<const IntAction> action,
                ThtsEnvContext& ctx) const 
            {
                Eigen::ArrayXd r = Eigen::ArrayXd::Zero(2);
                if (add_extra_rewards) {
                    r = Eigen::ArrayXd::Zero(4);
                }
                r[RIGHT] = -1.0; 
                r[DOWN] = -1.0; 
                // add bonus in r[dir], and add more if dir is different to last action
                r[action->action] += (std::get<2>(state->state) == action->action) ? same_dir_bonus : new_dir_bonus;
                if (!add_extra_rewards) {
                    return r;
                }
                
                if (action->action == RIGHT) {
                    r[2] = pow(gamma,std::get<0>(state->state));
                } else if (action->action == DOWN) {
                    r[3] = pow(gamma,std::get<1>(state->state));
                }
                return r;
            }

        /**
         * Interface implementation (basically calls the above implementations with surrounding casts).
         */
        public:
            virtual shared_ptr<const State> get_initial_state_itfc() const override {
                shared_ptr<const Int3TupleState> init_state = get_initial_state();
                return static_pointer_cast<const State>(init_state);
            }

            virtual bool is_sink_state_itfc(shared_ptr<const State> state, ThtsEnvContext& ctx) const {
                shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
                return is_sink_state(state_itfc, ctx);
            }

            virtual shared_ptr<ActionVector> get_valid_actions_itfc(
                shared_ptr<const State> state, ThtsEnvContext& ctx) const override
            {
                shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
                shared_ptr<IntActionVector> valid_actions_itfc = get_valid_actions(state_itfc, ctx);

                shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
                for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
                    valid_actions->push_back(static_pointer_cast<const Action>(act));
                }
                return valid_actions;
            }

            virtual shared_ptr<StateDistr> get_transition_distribution_itfc(
                shared_ptr<const State> state, shared_ptr<const Action> action, ThtsEnvContext& ctx) const override
            {
                shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<Int3TupleStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc, ctx);
                
                shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
                for (pair<shared_ptr<const Int3TupleState>,double> key_val_pair : *distr_itfc) {
                    shared_ptr<const State> state = static_pointer_cast<const State>(key_val_pair.first);
                    double prob = key_val_pair.second;
                    distr->insert_or_assign(state, prob);
                }
                return distr;
            }

            virtual shared_ptr<const State> sample_transition_distribution_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action, 
                RandManager& rand_manager, 
                ThtsEnvContext& ctx) const override 
            {
                shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                shared_ptr<const Int3TupleState> next_state = sample_transition_distribution(
                    state_itfc, action_itfc, rand_manager, ctx);
                return static_pointer_cast<const State>(next_state);
            }

            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                ThtsEnvContext& ctx) const override
            {
                return thts::ThtsEnv::get_observation_distribution_itfc(action, next_state, ctx);
            }

            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                RandManager& rand_manager, 
                ThtsEnvContext& ctx) const override 
            {
                return thts::ThtsEnv::sample_observation_distribution_itfc(action, next_state, rand_manager, ctx);
            }

            virtual Eigen::ArrayXd get_mo_reward_itfc(
                shared_ptr<const State> state, 
                shared_ptr<const Action> action,
                ThtsEnvContext& ctx) const override
            {
                shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
                shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
                return get_mo_reward(state_itfc, action_itfc, ctx);
            }
    };
}