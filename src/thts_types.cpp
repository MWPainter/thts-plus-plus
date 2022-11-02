#include "thts_types.h"

#include "helper_templates.h"

#include <functional> 
#include <iostream>
#include <sstream>

using namespace std;
using namespace thts;




namespace thts {
    /**
     * Implementation of virtual hash function for IntState
     */
    size_t IntState::hash() const {
        return std::hash<int>()(state);
    }

    /**
     * Implementation of virtual equals_itfc function for IntPairState
     */
    bool IntState::equals_itfc(const Observation& other) const {
        try {
            const IntState& oth = dynamic_cast<const IntState&>(other);
            return equals(oth);
        }
        catch (bad_cast) {
            return false;
        }
    }

    /**
     * Implementation of equals function for IntState
     */
    bool IntState::equals(const IntState& other) const {
        return state == other.state;
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    string IntState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << state << ")";
        return ss.str();
    }
    
    /**
     * Implementation of virtual hash function for IntPairState
     */
    size_t IntPairState::hash() const {
        size_t cur_hash = 0;
        cur_hash = helper::hash_combine(cur_hash,state.first);
        return helper::hash_combine(cur_hash, state.second);
    }

    /**
     * Implementation of virtual equals_itfc function for IntState
     */
    bool IntPairState::equals_itfc(const Observation& other) const {
        try {
            const IntPairState& oth = dynamic_cast<const IntPairState&>(other);
            return equals(oth);
        }
        catch (bad_cast) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for IntPairState
     */
    bool IntPairState::equals(const IntPairState& other) const {
        return state.first == other.state.first && state.second == other.state.second;
    }

    /**
     * Implementation of virtual equals function for IntPairState
     */
    string IntPairState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << state.first << "," << state.second << ")";
        return ss.str();
    }

    /**
     * Implementation of virtual hash function for IntAction
     */
    size_t IntAction::hash() const {
        return std::hash<int>()(action);
    }

    /**
     * Implementation of virtual equals_itfc function for IntAction
     */
    bool IntAction::equals_itfc(const Action& other) const {
        try {
            const IntAction& oth = dynamic_cast<const IntAction&>(other);
            return equals(oth);
        }
        catch (bad_cast) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    bool IntAction::equals(const IntAction& other) const {
        return action == other.action;
    }

    /**
     * Implementation of virtual equals function for IntState
     */
    string IntAction::get_pretty_print_string() const {
        stringstream ss;
        ss << action;
        return ss.str();
    }

    /**
     * Implementation of virtual hash function for StringAction
     */
    size_t StringAction::hash() const {
        return std::hash<string>()(action);
    }

    /**
     * Implementation of virtual equals_itfc function for StringAction
     */
    bool StringAction::equals_itfc(const Action& other) const {
        try {
            const StringAction& oth = dynamic_cast<const StringAction&>(other);
            return equals(oth);
        }
        catch (bad_cast) {
            return false;
        }
    }

    /**
     * Implementation of virtual equals function for StringAction
     */
    bool StringAction::equals(const StringAction& other) const {
        return action == other.action;
    }

    /**
     * Implementation of virtual equals function for StringAction
     */
    string StringAction::get_pretty_print_string() const {
        stringstream ss;
        ss << action;
        return ss.str();
    }
}

/**
 * Implementation of hash, equal_to and << operator functions for State/Action/Observation. Using virtual functions 
 * so that subclasses only need to implement those sub functions
 */
namespace std {
    /**
     * Implementation of std::hash<Observation>, calling the virtual hash function.
     */
    template <>
    struct hash<Observation> {
        size_t operator()(const Observation& observation) const {
            return observation.hash();
        }
    };
    template <>
    struct hash<shared_ptr<const Observation>> {
        size_t operator()(const shared_ptr<const Observation>& observation) const {
            return observation->hash();
        }
    };

    /**
     * Implementation of std::equal_to<Observation>, calling the equals function.
     */
    inline bool operator==(const Observation& lhs, const Observation& rhs) {
        return lhs.equals_itfc(rhs);
    }
    template <>
    struct equal_to<Observation> {
        bool operator()(const Observation& lhs, const Observation& rhs) const {
            return lhs.equals_itfc(rhs);
        }
    };
    inline bool operator==(const shared_ptr<const Observation>& lhs, const shared_ptr<const Observation>& rhs) {
        return lhs->equals_itfc(*rhs);
    }
    template <>
    struct equal_to<shared_ptr<const Observation>> {
        bool operator()(const shared_ptr<const Observation>& lhs, const shared_ptr<const Observation>& rhs) const {
            return lhs->equals_itfc(*rhs);
        }
    };

    /**
     * Override output stream << operator for Observation, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const Observation& observation) {
        os << observation.get_pretty_print_string();
        return os;
    }
    ostream& operator<<(ostream& os, const shared_ptr<const Observation>& observation) {
        os << observation->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<State>, calling the virtual hash function.
     */
    template <>
    struct hash<State> {
        size_t operator()(const State& state) const {
            return state.hash();
        }
    };
    template <>
    struct hash<shared_ptr<const State>> {
        size_t operator()(const shared_ptr<const State>& state) const {
            return state->hash();
        }
    };
    /**
     * Implementation of std::equal_to<State>, calling the equals function.
     */
    inline bool operator==(const State& lhs, const State& rhs) {
        return lhs.equals_itfc(rhs);
    }
    template <>
    struct equal_to<State> {
        bool operator()(const State& lhs, const State& rhs) const {
            return lhs.equals_itfc(rhs);
        }
    };
    inline bool operator==(const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs) {
        return lhs->equals_itfc(*rhs);
    }
    template <>
    struct equal_to<shared_ptr<const State>> {
        bool operator()(const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs) const {
            return lhs->equals_itfc(*rhs);
        }
    };

    /**
     * Override output stream << operator for State, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const State& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    ostream& operator<<(ostream& os, const shared_ptr<const State>& state) {
        os << state->get_pretty_print_string();
        return os;
    }

    /**
     * Implementation of std::hash<Action>, calling the virtual hash function.
     */
    template <>
    struct hash<Action> {
        size_t operator()(const Action& action) const {
            return action.hash();
        }
    };
    template <>
    struct hash<shared_ptr<const Action>> {
        size_t operator()(const shared_ptr<const Action>& action) const {
            return action->hash();
        }
    };

    /**
     * Implementation of std::equal_to<Action>, calling the equals function.
     */
    inline bool operator==(const Action& lhs, const Action& rhs) {
        return lhs.equals_itfc(rhs);
    }
    template <>
    struct equal_to<Action> {
        bool operator()(const Action& lhs, const Action& rhs) const {
            return lhs.equals_itfc(rhs);
        }
    };
    inline bool operator==(const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs) {
        return lhs->equals_itfc(*rhs);
    }
    template <>
    struct equal_to<shared_ptr<const Action>> {
        bool operator()(const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs) const {
            return lhs->equals_itfc(*rhs);
        }
    };

    /**
     * Override output stream << operator for Action, using get_pretty_print_string function.
     */
    ostream& operator<<(ostream& os, const Action& action) {
        os << action.get_pretty_print_string();
        return os;
    }
    ostream& operator<<(ostream& os, const shared_ptr<const Action>& action) {
        os << action->get_pretty_print_string();
        return os;
    }
}

/**
 * Implementation of << operator for common typedefs
 */
namespace std {

    ostream& operator<<(ostream& os, const ActionVector& vec) {
        os << helper::vector_pretty_print_string(vec);
        return os;
    }

    ostream& operator<<(ostream& os, const StringActionVector& vec) {
        os << helper::vector_pretty_print_string(vec);
        return os;
    }

    ostream& operator<<(ostream& os, const StateDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }

    ostream& operator<<(ostream& os, const ObservationDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }

    ostream& operator<<(ostream& os, const IntPairStateDistr& distr) {
        os << helper::unordered_map_pretty_print_string(distr);
        return os;
    }
}

/**
 * Implementation of hash, equals_to and output stream functions for transposition table types
 */
namespace std {
    /**
     * Implementation of std::hash<DNodeIdTuple>.
     */
    template <>
    struct hash<DNodeIdTuple> {
        size_t operator()(const DNodeIdTuple& tpl) const {
            size_t hash_val = 0;
            hash_val = helper::hash_combine(hash_val, get<0>(tpl));
            hash_val = helper::hash_combine(hash_val, get<1>(tpl));
            return hash_val;
        }
    };

    /**
     * Implementation of std::equal_to<DNodeIdTuple>.
     */
    template <>
    struct equal_to<DNodeIdTuple> {
        bool operator()(const DNodeIdTuple& lhs, const DNodeIdTuple& rhs) const {
            return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs);
        }
    };

    /**
     * Override output stream << operator for DNodeIdTuple.
     */
    ostream& operator<<(ostream& os, const DNodeIdTuple& tpl) {
        os << "DNodeId(" << get<0>(tpl) << "," << get<1>(tpl) << ")";
        return os;
    }

    /**
     * Implementation of std::hash<CNodeIdTuple>.
     */
    template <>
    struct hash<CNodeIdTuple> {
        size_t operator()(const CNodeIdTuple& tpl) const {
            size_t hash_val = 0;
            hash_val = helper::hash_combine(hash_val, get<0>(tpl));
            hash_val = helper::hash_combine(hash_val, get<1>(tpl));
            hash_val = helper::hash_combine(hash_val, get<2>(tpl));
            return hash_val;
        }
    };

    /**
     * Implementation of std::equal_to<CNodeIdTuple>.
     */
    template <>
    struct equal_to<CNodeIdTuple> {
        bool operator()(const CNodeIdTuple& lhs, const CNodeIdTuple& rhs) const {
            return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs) && get<2>(lhs) == get<2>(rhs);
        }
    };

    /**
     * Override output stream << operator for CNodeIdTuple.
     */
    ostream& operator<<(ostream& os, const CNodeIdTuple& tpl) {
        os << "CNodeId(" << get<0>(tpl) << "," << get<1>(tpl) << "," << get<2>(tpl) << ")";
        return os;
    }

    /**
     * Override output stream << operator for DNodeTable.
     */
    ostream& operator<<(ostream& os, const DNodeTable& tbl) {
        os << helper::unordered_map_pretty_print_string(tbl);
        return os;
    }

    /**
     * Override output stream << operator for CNodeTable.
     */
    ostream& operator<<(ostream& os, const CNodeTable& tbl) {
        os << helper::unordered_map_pretty_print_string(tbl);
        return os;
    }
}