// #include "thts_manager.h"

// #include "helper_templates.h"

// using namespace thts;

// /**
//  * Implementation of hash, equals_to and output stream functions for transposition table types
//  */
// namespace std {
//     /**
//      * Implementation of std::hash<DNodeIdTuple>.
//      */
//     template <>
//     struct hash<DNodeIdTuple> {
//         size_t operator()(const DNodeIdTuple& tpl) const {
//             size_t hash_val = 0;
//             hash_val = helper::hash_combine(hash_val, get<0>(tpl));
//             hash_val = helper::hash_combine(hash_val, get<1>(tpl));
//             return hash_val;
//         }
//     };

//     /**
//      * Implementation of std::equal_to<DNodeIdTuple>.
//      */
//     template <>
//     struct equal_to<DNodeIdTuple> {
//         bool operator()(const DNodeIdTuple& lhs, const DNodeIdTuple& rhs) const {
//             return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs);
//         }
//     };

//     /**
//      * Override output stream << operator for DNodeIdTuple.
//      */
//     ostream& operator<<(ostream& os, const DNodeIdTuple& tpl) {
//         os << "DNodeId(" << get<0>(tpl) << "," << get<1>(tpl) << ")";
//         return os;
//     }

//     /**
//      * Implementation of std::hash<CNodeIdTuple>.
//      */
//     template <>
//     struct hash<CNodeIdTuple> {
//         size_t operator()(const CNodeIdTuple& tpl) const {
//             size_t hash_val = 0;
//             hash_val = helper::hash_combine(hash_val, get<0>(tpl));
//             hash_val = helper::hash_combine(hash_val, get<1>(tpl));
//             hash_val = helper::hash_combine(hash_val, get<2>(tpl));
//             return hash_val;
//         }
//     };

//     /**
//      * Implementation of std::equal_to<CNodeIdTuple>.
//      */
//     template <>
//     struct equal_to<CNodeIdTuple> {
//         bool operator()(const CNodeIdTuple& lhs, const CNodeIdTuple& rhs) const {
//             return get<0>(lhs) == get<0>(rhs) && get<1>(lhs) == get<1>(rhs) && get<2>(lhs) == get<2>(rhs);
//         }
//     };

//     /**
//      * Override output stream << operator for CNodeIdTuple.
//      */
//     ostream& operator<<(ostream& os, const CNodeIdTuple& tpl) {
//         os << "CNodeId(" << get<0>(tpl) << "," << get<1>(tpl) << "," << get<2>(tpl) << ")";
//         return os;
//     }

//     /**
//      * Override output stream << operator for DNodeTable.
//      */
//     ostream& operator<<(ostream& os, const DNodeTable& tbl) {
//         os << helper::unordered_map_pretty_print_string(tbl);
//         return os;
//     }

//     /**
//      * Override output stream << operator for CNodeTable.
//      */
//     ostream& operator<<(ostream& os, const CNodeTable& tbl) {
//         os << helper::unordered_map_pretty_print_string(tbl);
//         return os;
//     }
// }