#include "thts_manager.h"

using namespace std;

namespace thts {


    /**
     * Constructor. Initialises values directly other than random number generation.
     */    
    ThtsManager::ThtsManager(const ThtsManagerArgs& args) : 
        RandManager(args.seed),
        thts_envs(),
        num_threads(args.num_threads),
        num_envs(args.num_envs),
        max_depth(args.max_depth),
        heuristic_fn(args.heuristic_fn),
        prior_fn(args.prior_fn),
        mcts_mode(args.mcts_mode), 
        use_transposition_table(args.use_transposition_table), 
        is_two_player_game(args.is_two_player_game),
        dmap_lock(),
        dmap(),
        thread_id_map_lock(),
        thread_id_map(),
        thts_context_map_lock(),
        thts_context_map((args.thts_env != nullptr) ? args.thts_env->sample_context_and_reset_itfc(0) : make_shared<ThtsEnvContext>()) 
    {
        thts_envs.push_back(args.thts_env);
        for (int i=1; i<num_envs; i++) {
            thts_envs.push_back(args.thts_env->clone());
        }
    }
    
    /**
     * Get a thts env from the vector, without knowing the thread id
    */
    std::shared_ptr<ThtsEnv> ThtsManager::thts_env() {
        return thts_env(get_thts_thread_id());
    }
    
    /**
     * Get a thts env from the vector
    */
    std::shared_ptr<ThtsEnv> ThtsManager::thts_env(int tid) {
        return thts_envs[tid % num_envs];
    }

    /**
     * Register the thread calling this function with thts thread id 'tid'
    */
    void ThtsManager::register_thread_id(int tid) {
        unique_lock<shared_mutex> writer_lg(thread_id_map_lock);
        thread_id_map[std::this_thread::get_id()] = tid;
    }

    /**
     * Get the thts thread id of the current thread
    */
    int ThtsManager::get_thts_thread_id() {
        shared_lock<shared_mutex> reader_lg(thread_id_map_lock);
        return thread_id_map[std::this_thread::get_id()];
    }

    /**
     * Register the thts context 'ctx' with thts thread id 'tid'
    */
    void ThtsManager::register_thts_context(int tid, std::shared_ptr<ThtsEnvContext> ctx) {
        unique_lock<shared_mutex> writer_lg(thts_context_map_lock);
        thts_context_map[tid] = ctx;
    }

    /**
     * Gets the current thts context for the calling thread
    */
    std::shared_ptr<ThtsEnvContext> ThtsManager::get_thts_context() {
        int tid = get_thts_thread_id();
        return get_thts_context(tid);
    }


    /**
     * Gets the current thts context for the calling thread
    */
    std::shared_ptr<ThtsEnvContext> ThtsManager::get_thts_context(int tid) {
        shared_lock<shared_mutex> reader_lg(thts_context_map_lock);
        return thts_context_map[tid];
    }
}