#pragma once

#include "main_aux/envs/d_chain.h"

namespace thts {

    /** 
     * Implementation of DChain with entropy trap with variable lengths and end reward
     */
    class EntropyTrapEnv : public DChainEnv {

        /**
         * Core EntropyTrapEnv implementaion.
         * 
         * 
         * Like the DChain env, with an additional chain at the end to trap entropy maximising agents. To see the 
         * original MDP see the paper / thesis.
         * 
         * This implementation is a bit simpler, and will be adapted from the Dchain implementation:
         * 0 -------> 1 -------> 2 -> ... -> D-2 ------> D-1 --- (R=1) ---> D 
         * |          |          |            |           |
         * R=D-1/D    R=D-2/D    R=D-3/D      R=1/D       R=0
         * |          |          |            |           |
         * -1         -1         -1           -1          -1
         * 
         * Now at state D, there is a choice between taking the entropy trap or the max reward. In the paper/thesis we 
         * depict this using a sequence of pairs of states, but it doesn't really need to be pairs of states, just 
         * that there are two actions to continue down the entropy trap part of the chain:
         * 0 -------> 1 -------> 2 -> ... -> D-2 ------> D-1 -------> D ------> D+1 -----> D+2 -----> D+3 -> ... -> D+H -----> D+H+1
         * |          |          |            |           |           |          |_________^ |________^ |___ ... ___^ |_________^
         * R=D-1/D    R=D-2/D    R=D-3/D      R=1/D       R=0         R=1
         * |          |          |            |           |           |
         * -1         -1         -1           -1          -1          -1
         * 
         * To implement this we just need to override a handful of functions from the d_chain env
         * 
         * Member variables:
         *      H: The length of the entropy trap portion
         */
        protected:
            int H;



        /**
         * Core ThtsEnv implementation functinos.
         */
        public:
            /**
             * Constructor
             */
            EntropyTrapEnv(int D, int H, double final_reward);

            virtual std::shared_ptr<ThtsEnv> clone() override;

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~EntropyTrapEnv() = default;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            virtual bool is_sink_state(std::shared_ptr<const IntState> state) const override;

            /**
             * Samples an successor state when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible successor state that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an successor state sampled from taking 'action' from 'state'
             */
            virtual std::shared_ptr<const IntState> sample_transition_distribution(
                std::shared_ptr<const IntState> state, 
                std::shared_ptr<const IntAction> action) const override;
            
            /**
             * Returns the reward for a given state, action, observation tuple.
             * 
             * Commonly the reward is written as a function of just the state and action pair. But we provide the 
             * option to depend on the observation too. 
             * 
             * Args:
             *      state: The current state to get a reward for
             *      action: The action taken to get a reward for
             *      observation: 
             *          The (optional) observation sampled from the state, action pair that can optionally be used as 
             *          part of the reward function.
             * 
             * Returns:
             *      The reward for taking 'action' from 'state' (and sampling 'observation')
             */
            virtual double get_reward(
                std::shared_ptr<const IntState> state, 
                std::shared_ptr<const IntAction> action) const override;
    };
}