#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

/**
 * thts_types.h
 * 
 * This file contains some base types used by thts, namely State, Action and Observation classes.
 * 
 * These implementations are just empty classes which serve as a supertype to be used in Thts classes.
 * The State class inherits from the Observation class so that we can return a State as an Observation, as Observations 
 * can be the state in fully observable environments. 
 * 
 * State, Action and Observation objects need to be used as keys in unordered_map objects. Which means that they need 
 * to implement the relevent std::hash<State>, std::equal_to<State> and std::hash<Action>, std::equal_to<Action> and 
 * std::hash<Observation>, std::equal_to<Observation> classes.
 */

namespace thts {

    /**
     * A abstract base type to use for Observations.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In thts_types.cpp, we 
     * implement the std::hash<Observation> and std::equal_to<Observation> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     */
    class Observation {
        public:
            virtual std::size_t hash() const = 0;
            virtual bool equals_itfc(const Observation& other) const = 0;
            virtual std::string get_pretty_print_string() const = 0;
    };



    /**
     * A abstract base type to use for States.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In thts_types.cpp, we 
     * implement the std::hash<State> and std::equal_to<State> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     */
    class State : public Observation {
        public:
            virtual std::size_t hash() const = 0;
            virtual bool equals_itfc(const Observation& other) const = 0;
            virtual std::string get_pretty_print_string() const = 0;
    };
    


    /**
     * A abstract base type to use for Actions.
     * 
     * Virtual functions are used to provide hash, equality and printing functionality. In thts_types.cpp, we 
     * implement the std::hash<Action> and std::equal_to<Action> classes using these virtual functions, 
     * as well as the operator<< for ostreams.
     */
    class Action {
        public:
            virtual std::size_t hash() const = 0;
            virtual bool equals_itfc(const Action& other) const = 0;
            virtual std::string get_pretty_print_string() const = 0;
    };



    /**
     * An implementaton of state containing a single integer state.
     */
    class IntState : public State {
        public:
            int state;

            IntState(int state) : state(state) {}
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Observation& other) const;
            bool equals(const IntState& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * An implementaton of state containing a pair of integers as the state.
     */
    class IntPairState : public State {
        public:
            std::pair<int,int> state;

            IntPairState(std::pair<int,int> pr) : state(pr) {}
            IntPairState(int first, int second) : state(std::make_pair(first,second)) {}
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Observation& other) const;
            bool equals(const IntPairState& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * An implementation of action containing a single int as an action
     */
    class IntAction : public Action {
        public:
            int action;

            IntAction(int action) : action(action) {}
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Action& other) const;
            bool equals(const IntAction& other) const;
            virtual std::string get_pretty_print_string() const;
    };

    /**
     * An implementation of action containing a single string as an action
     */
    class StringAction : public Action {
        public:
            std::string action;

            StringAction(std::string action) : action(action) {}
            virtual std::size_t hash() const;
            virtual bool equals_itfc(const Action& other) const;
            bool equals(const StringAction& other) const;
            virtual std::string get_pretty_print_string() const;
    };



    /**
     * Typedef for heuristic function pointers
     * First used in thts_decision_node.h and thts_chance_node.h
     */  
    typedef double (*HeuristicFnPtr) (std::shared_ptr<const State>, std::shared_ptr<const Action>);

    /**
     * Typedef for (action) prior function pointers
     * First used in thts_decision_node.h and thts_chance_node.h
     */
    typedef std::unordered_map<Action,double> (*PriorFnPtr) (std::shared_ptr<const State>); //, shared_ptr<const Action>);



    /**
     * Commonly want a list of actions, and the type is a bit verbose.
     */
    typedef std::vector<std::shared_ptr<const Action>> ActionVector;
    typedef std::vector<std::shared_ptr<const StringAction>> StringActionVector;

    /**
     * Also commonly want a distribution over observations/states, and the type is also verbose.
     */
    typedef std::unordered_map<std::shared_ptr<const State>,double> StateDistr;
    typedef std::unordered_map<std::shared_ptr<const Observation>,double> ObservationDistr;
    typedef std::unordered_map<std::shared_ptr<const IntPairState>,double> IntPairStateDistr;



    // forward declare
    class ThtsDNode;
    class ThtsCNode;

    /**
     * Typedef for dnode id tuples, for readability.
     * First used transposition table implementation, in thts_manager.h, thts_decision_node.h and thts_chance_node.h
     */
    typedef std::tuple<int,std::shared_ptr<const Observation>> DNodeIdTuple;
    typedef std::unordered_map<DNodeIdTuple,std::shared_ptr<ThtsDNode>> DNodeTable;

    /**
     * Typedef for cnode id tuples, for readability.
     * First used transposition table implementation, in thts_manager.h, thts_decision_node.h and thts_chance_node.h
     */
    typedef std::tuple<int,std::shared_ptr<const State>,std::shared_ptr<const Action>> CNodeIdTuple;
    typedef std::unordered_map<CNodeIdTuple,std::shared_ptr<ThtsCNode>> CNodeTable;
}

/**
 * Forward declaring the hash, equality and output stream functions defined in thts_types.cpp.
 * Needed so other files know to look at thts_types.o to find implementations of these functions.
 */
namespace std {
    using namespace thts;

    /**
     * Hash, equality and output stream functins for Observation.
     */
    template <> struct hash<Observation>;
    template <> struct hash<shared_ptr<const Observation>>;
    
    inline bool operator==(const Observation& lhs, const Observation& rhs);
    template <> struct equal_to<Observation>;
    inline bool operator==(const shared_ptr<const Observation>& lhs, const shared_ptr<const Observation>& rhs);
    template <> struct equal_to<shared_ptr<const Observation>>;

    ostream& operator<<(ostream& os, const Observation& observation);
    ostream& operator<<(ostream& os, const shared_ptr<const Observation>& observation);

    /**
     * Hash, equality and output stream functins for State.
     */
    template <> struct hash<State>;
    template <> struct hash<shared_ptr<const State>>;
    
    inline bool operator==(const State& lhs, const State& rhs);
    template <> struct equal_to<State>;
    inline bool operator==(const shared_ptr<const State>& lhs, const shared_ptr<const State>& rhs);
    template <> struct equal_to<shared_ptr<const State>>;

    ostream& operator<<(ostream& os, const State& state);
    ostream& operator<<(ostream& os, const shared_ptr<const State>& state);

    /**
     * Hash, equality and output stream functins for Action.
     */
    template <> struct hash<Action>;
    template <> struct hash<shared_ptr<const Action>>;
    
    inline bool operator==(const Action& lhs, const Action& rhs);
    template <> struct equal_to<Action>;
    inline bool operator==(const shared_ptr<const Action>& lhs, const shared_ptr<const Action>& rhs);
    template <> struct equal_to<shared_ptr<const Action>>;

    ostream& operator<<(ostream& os, const Action& action);
    ostream& operator<<(ostream& os, const shared_ptr<const Action>& Action);

    /**
     * Output stream overloeads for common vector and map typedefs
     */ 
    ostream& operator<<(ostream& os, const ActionVector& vec);
    ostream& operator<<(ostream& os, const StringActionVector& vec);
    ostream& operator<<(ostream& os, const StateDistr& distr);
    ostream& operator<<(ostream& os, const ObservationDistr& distr);
    ostream& operator<<(ostream& os, const IntPairStateDistr& distr);

    /**
     * Hash, equality and stream functions for DNodeIdTuple
     */
    template <> struct hash<DNodeIdTuple>;
    template <> struct equal_to<DNodeIdTuple>;
    ostream& operator<<(ostream& os, const DNodeIdTuple& tpl);

    /**
     * Hash, equality and stream functions for CNodeIdTuple
     */
    template <> struct hash<CNodeIdTuple>;
    template <> struct equal_to<CNodeIdTuple>;
    ostream& operator<<(ostream& os, const CNodeIdTuple& tpl);

    /**
     * Output streams for transposition tables
     */
    ostream& operator<<(ostream& os, const DNodeTable& tbl);
    ostream& operator<<(ostream& os, const CNodeTable& tbl);
}