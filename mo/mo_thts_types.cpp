#include "mo/mo_thts_types.h"

#include "helper_templates.h"

using namespace std;


namespace thts {
    
    /**
     * Implementation of virtual hash function for IntVectorState
     */
    size_t IntVectorState::hash() const {
        size_t cur_hash = 0;
        for (size_t i=0; i < state.size(); i++) {
            cur_hash = helper::hash_combine(cur_hash,state.at(i));
        }
        return cur_hash;
    }

    /**
     * Implementation of virtual equals_itfc function for IntVectorState
     */
    bool IntVectorState::equals_itfc(const Observation& other) const {
        try {
            const IntVectorState& oth = dynamic_cast<const IntVectorState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for IntVectorState
     */
    bool IntVectorState::equals(const IntVectorState& other) const {
        if (state.size() != other.state.size()) {
            return false;
        }
        for (size_t i=0; i<state.size(); i++) {
            if (state.at(i) != other.state.at(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Implementation of virtual equals function for IntVectorState
     */
    string IntVectorState::get_pretty_print_string() const {
        stringstream ss;
        ss << "("; 
        for (size_t i=0; i<state.size(); i++) {
            ss << state.at(i) << ",";
        }        
        ss << ")";
        return ss.str();
    }


}




namespace std {

    ostream& operator<<(ostream& os, const IntVectorState& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const IntVectorState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }


}