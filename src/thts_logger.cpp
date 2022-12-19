#include "thts_logger.h"

using namespace std;

/**
 * Logger entry default implementation
*/
namespace thts {
    LoggerEntry::LoggerEntry(chrono::duration<double> runtime, int num_visits) : 
        runtime(runtime), num_visits(num_visits) {}

    void LoggerEntry::write_header_to_ostream(ostream& os) {
        os << "runtime" << ",";
        os << "num_visits";
    }

    void LoggerEntry::write_to_ostream(ostream& os) {
        os << runtime.count() << ",";
        os << num_visits;
    }
}

/**
 * Logger default implementation
*/
namespace thts {

    ThtsLogger::ThtsLogger() : 
        prior_runtime(chrono::duration<double>::zero()), 
        trials_delta(numeric_limits<int>::max()), 
        runtime_delta(numeric_limits<double>::max()) {}

    void ThtsLogger::set_trials_delta(int delta) {
        trials_delta = delta;
    }
    
    void ThtsLogger::set_runtime_delta(double delta) {
        runtime_delta = chrono::duration<double>(delta);
    }
    
    int ThtsLogger::size() const {
        return entries.size();
    }
    
    void ThtsLogger::add_origin_entry() {
        entries.push_back(LoggerEntry(
            chrono::duration<double>::zero(), 
            0));
    }
    
    void ThtsLogger::reset_start_time() {
        start_time = chrono::system_clock::now();
        last_log_num_trials = 0;
        next_log_runtime_threshold = runtime_delta;
    }
    
    chrono::duration<double> ThtsLogger::get_current_total_runtime() {
        chrono::duration<double> cur_runtime = chrono::system_clock::now() - start_time;
        return prior_runtime + cur_runtime;
    }
    
    bool ThtsLogger::should_log(int num_trials) {
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
    
    void ThtsLogger::log(shared_ptr<ThtsDNode> node) {
        entries.push_back(LoggerEntry(
            get_current_total_runtime(), 
            node->num_visits));
    }
    
    void ThtsLogger::update_prior_runtime() {
        chrono::duration<double> runtime = chrono::system_clock::now() - start_time;
        prior_runtime += runtime;
    }

    void ThtsLogger::write_to_ostream(ostream& os) {
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
}