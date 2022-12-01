#pragma once

#include "thts_decision_node.h"

#include <chrono>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>

namespace thts {
    /**
     * Base class for entries in logger
     * 
     * Member variables:
     *      runtime: The total runtime for this datapoint
     *      num_trials: The number of trials completed for this datapoint
     */
    struct LoggerEntry {
        std::chrono::duration<double> runtime;
        int num_trials;

        LoggerEntry(std::chrono::duration<double> runtime, int num_trials) : 
            runtime(runtime), num_trials(num_trials) {}

        /**
         * Writes a header for an ostream, so we would know what the entries correspond to in a csv
         * 
         * Args:
         *      os: An output stream
         */
        virtual void write_header_to_ostream(std::ostream& os) {
            os << "runtime" << ",";
            os << "num_trials";
        }

        /**
         * Writes this entry to an ostream
         * 
         * Args:
         *      os: An output stream
         */
        virtual void write_to_ostream(std::ostream& os) {
            os << runtime.count() << ",";
            os << num_trials;
        }
    };

    /**
     * Abstract logger class
     * 
     * Member variables:
     *      entries: A vector of LoggerEntry objects, each representing a datatype
     *      prior_runtime: The amount of runtime logger from previous calls to ThtsPool run_trials
     *      start_time: The time at the start of the last run_trials call in ThtsPool
     *      trials_delta: 
     *          Indicates 'log' should be called every 'trials_delta' completed trials.
     *      last_log_num_trials:
     *          The value of 'num_trials' the last time 'log' was (should have been) called
     *      runtime_delta: 
     *          Indicates 'log' should be called every 'runtime_delta' seconds. Default is max value which means never 
     *          log because of runtime.
     *      next_log_runtime_threshold: 
     *          The next runtime duration that we should log at 
     */
    class ThtsLogger {
        protected:
            std::vector<LoggerEntry> entries;
            std::chrono::duration<double> prior_runtime;
            std::chrono::time_point<std::chrono::system_clock> start_time;

            int trials_delta;
            int last_log_num_trials;
            std::chrono::duration<double> runtime_delta;
            std::chrono::duration<double> next_log_runtime_threshold;

        public:
            ThtsLogger() : 
                prior_runtime(0.0), 
                trials_delta(std::numeric_limits<int>::max()), 
                runtime_delta(std::numeric_limits<double>::max()) {}

            virtual ~ThtsLogger() = default;

            /**
             * Setter for trials delta
             */
            void set_trials_delta(int delta) {
                trials_delta = delta;
            }

            /**
             * Setter for runtime delta
             */
            void set_runtime_delta(double delta) {
                runtime_delta = std::chrono::duration<double>(delta);
            }

            /**
             * Gets the size of the logger
             */
            int size() const {
                return entries.size();
            }

            /**
             * Adds an entry to 'entries' that represents an origin point
             */
            virtual void add_origin_entry() = 0;

            /**
             * Call this at the beginning of a run_trials call to set start time 
             */
            virtual void reset_start_time() {
                start_time = std::chrono::system_clock::now();
                last_log_num_trials = 0;
                next_log_runtime_threshold = runtime_delta;
            }

            /**
             * Helper to get the current runtime
             * 
             * Returns:
             *      The total runtime used in the thts routine so far
             */
            std::chrono::duration<double> get_current_total_runtime() {
                std::chrono::duration<double> cur_runtime = std::chrono::system_clock::now() - start_time;
                return prior_runtime + cur_runtime;
            }

            /**
             * Checks if it is time to call log
             * 
             * Note that if the deltas are at their default values, the rhs of the comparisons will always be a max 
             * value and the checks will never pass
             * 
             * Args:
             *      num_trials: The number of trials run in the current 'run_trials' call to ThtsPool
             * 
             * Returns:
             *      If 'log' should be called for this current trial
             */
            bool should_log(int num_trials) {
                bool result = false;
                if (num_trials >= last_log_num_trials + trials_delta) {
                    last_log_num_trials = num_trials;
                    result = true;
                }
                if (get_current_total_runtime() >= next_log_runtime_threshold) {
                    next_log_runtime_threshold += runtime_delta;
                    result = true;
                }
                return result;
            }

            /**
             * Adds an logger entry based off the current 
             * 
             * Assumes that the lock for the node has already been 
             * 
             * Args:
             *      node: A (root) node to log information about
             */
            virtual void log(std::shared_ptr<ThtsDNode> node) = 0;

            /**
             * Call this when all trials have been run
             */
            virtual void update_prior_runtime() {
                std::chrono::duration<double> runtime = std::chrono::system_clock::now() - start_time;
                prior_runtime += runtime;
            }

            /**
             * Write logger to an ostream
             * 
             * Args:
             *      os: The output stream to write to
             */
            virtual void write_to_ostream(std::ostream& os) {
                if (entries.size() == 0) return;
                LoggerEntry& first_entry = entries[0];
                first_entry.write_header_to_ostream(os);
                os << "\n";

                for (LoggerEntry& logger_entry : entries) {
                    logger_entry.write_to_ostream(os);
                    os << "\n";
                }

                os.flush();
            }
    };
}