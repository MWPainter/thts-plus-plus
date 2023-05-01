#include "algorithms/uct/uct_logger.h"

using namespace std;

/**
 * Logger entry default implementation
*/
namespace thts {
    UctLoggerEntry::UctLoggerEntry(
        chrono::duration<double> runtime, int num_visits, int num_backups, double avg_return) : 
            LoggerEntry(runtime, num_visits), num_backups(num_backups), avg_return(avg_return) {}

    void UctLoggerEntry::write_header_to_ostream(ostream& os) {
        os << "runtime," 
            << "num_visits," 
            << "num_backups,"
            << "avg_return";
    }

    void UctLoggerEntry::write_to_ostream(ostream& os) {
        os << runtime.count() << ","
            << num_visits << ","
            << num_backups << "," 
            << avg_return;
    }
}

/**
 * Logger default implementation
*/
namespace thts {

    UctLogger::UctLogger() : 
        ThtsLogger() {}
    
    void UctLogger::add_origin_entry() {
        entries.push_back(UctLoggerEntry(
            chrono::duration<double>::zero(), 
            0,
            0,
            0.0));
    }
    
    void UctLogger::log(shared_ptr<ThtsDNode> node) {
        UctDNode& uct_node = (UctDNode&) *node;
        entries.push_back(UctLoggerEntry(
            get_current_total_runtime(), 
            uct_node.num_visits,
            uct_node.num_backups,
            uct_node.avg_return));
    }
}