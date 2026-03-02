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



/**
 * Implementation of Vec
 */
namespace thts {

    Vec::Vec(const Eigen::ArrayXd& v) : 
        vec() 
    {
        // Apparently eigen does some fast stuff and better not to init in initialiser list
        // https://stackoverflow.com/questions/47644021/eigen-copy-constructor-vs-operator-performance
        vec = v;
    }

    Vec::Vec(Eigen::ArrayXd&& v) : 
        vec() 
    {
        // Apparently eigen does some fast stuff and better not to init in initialiser list
        // https://stackoverflow.com/questions/47644021/eigen-copy-constructor-vs-operator-performance
        vec = std::move(v);
    }

    Vec::Vec(const std::vector<double>& v) : 
        vec() 
    {
        // Apparently eigen does some fast stuff and better not to init in initialiser list
        // https://stackoverflow.com/questions/47644021/eigen-copy-constructor-vs-operator-performance

        // TODO: check if this makes a copy? Is this invalid if the vector is deallocated? what ive the values in the vector are changed?
        vec = Eigen::Map<const Eigen::ArrayXd>(v.data(), v.size());
        // vec = Eigen::ArrayXd(v.size());
        // for (size_t i=0; i<v.size(); i++) {
        //     vec[i] = v[i];
        // }
    }

    Vec::Vec(const Vec& other) : 
        vec() 
    {    
        // Apparently eigen does some fast stuff and better not to init in initialiser list
        // https://stackoverflow.com/questions/47644021/eigen-copy-constructor-vs-operator-performance
        vec = other.vec;
    }

    Vec::Vec(Vec&& other) : 
        vec() 
    {    
        // Apparently eigen does some fast stuff and better not to init in initialiser list
        // https://stackoverflow.com/questions/47644021/eigen-copy-constructor-vs-operator-performance
        vec = std::move(other.vec);
    }

    Vec Vec::Zero(int dim) {
        return Vec(Eigen::ArrayXd::Zero(dim));
    }

    Vec Vec::Const(int dim, double init_val) {
        return Vec(Eigen::ArrayXd::Constant(dim, init_val));
    }

    int Vec::size() const {
        return vec.size();
    }

    int Vec::dim() const {
        return vec.size();
    }

    double Vec::norm() const {
        return sqrt(vec.pow(2.0).sum());
    }

    Vec Vec::normalised() const {
        return Vec(this->vec / norm());
    }

    Vec Vec::project_onto_origin_line(const Vec& direction) const {
        Vec normalised_dir = direction.normalised();
        return normalised_dir * normalised_dir.dot(*this);
    }

    /**
     * To project onto arbitrary line, translate problem to origin, and then translate back after projection
     */
    Vec Vec::project_onto_line(const Vec& direction, const Vec& point) const {
        Vec translated_vec = *this - point;
        Vec translated_projected_vec = translated_vec.project_onto_origin_line(direction);
        return translated_projected_vec + point;
    }

    double Vec::dist(const Vec& other) const {
        return (*this - other).norm();
    }

    double Vec::dot(const Vec& other) const {
        return (this->vec * other.vec).sum();
    }

    bool Vec::weakly_pareto_dominates(const Vec& other) const
    {
        if (vec.size() != other.vec.size()) {
            throw runtime_error("Trying to use 'weakly_pareto_dominates' with vectors with different dims.");
        }
        return (vec >= other.vec).all();
    }

    bool Vec::strongly_pareto_dominates(const Vec& other, double tolerance) const
    {
        if (vec.size() != other.vec.size()) {
            throw runtime_error("Trying to use 'strongly_pareto_dominate' with vectors with different dims.");
        }
        return (vec + tolerance > other.vec).all();
    }

    bool Vec::equals(const Vec& other) const {
        if (vec.size() != other.vec.size()) {
            return false;
        }
        return (vec == other.vec).all();
    }

    bool Vec::approx_equals(const Vec& other, double tolerance) const {
        if (vec.size() != other.vec.size()) {
            return false;
        }
        // Use Eigen's isApprox for approximate equality
        return vec.isApprox(other.vec, tolerance);
    }

    std::size_t Vec::hash() const {
        size_t cur_hash = 0;
        cur_hash = thts::helper::hash_combine(cur_hash, vec.size());
        for (int i=0; i < vec.size(); i++) {
            cur_hash = helper::hash_combine(cur_hash,vec[i]);
        }
        return cur_hash;
    }




    Vec& Vec::operator+=(const double s) {
        vec += s;
        return *this;
    }

    Vec& Vec::operator*=(const double s) {
        vec *= s;
        return *this;
    }


    Vec Vec::operator+(const Vec& other) const {
        return Vec(vec + other.vec);
    }

    Vec Vec::operator-(const Vec& other) const {
        return Vec(vec - other.vec);
    }

    Vec Vec::operator*(const Vec& other) const {
        return Vec(vec * other.vec);
    }

    Vec Vec::operator/(const Vec& other) const {
        return Vec(vec / other.vec);
    }




    Vec& Vec::operator=(const Vec& other) {
        vec = other.vec;
        return *this;
    }
    
    Vec& Vec::operator=(Vec&& other) {
        vec = std::move(other.vec);
        return *this;
    }

    Vec& Vec::operator+=(const Vec& other) {
        vec += other.vec;
        return *this;
    }

    Vec& Vec::operator-=(const Vec& other) {
        vec -= other.vec;
        return *this;
    }

    Vec& Vec::operator*=(const Vec& other) {
        vec *= other.vec;
        return *this;
    }

    Vec& Vec::operator/=(const Vec& other) {
        vec /= other.vec;
        return *this;
    }



    bool Vec::operator==(const Vec& other) const {
        return equals(other);
    }

    bool Vec::operator!=(const Vec& other) const {
        return !equals(other);
    }



    double Vec::operator[](size_t i) const {
        return vec[i];
    }

}




namespace std {

    /**
     * OStream operator for IntVectorState
     */
    ostream& operator<<(ostream& os, const IntVectorState& state) {
        os << state.get_pretty_print_string();
        return os;
    }
    
    ostream& operator<<(ostream& os, const shared_ptr<const IntVectorState>& state) {
        os << state->get_pretty_print_string();
        return os;
    }


    /**
     * Vec Hash
    */
    size_t hash<Vec>::operator()(const Vec& vec) const {
        return vec.hash();
    }

    /**
     * Vec Equals
    */
    bool equal_to<Vec>::operator()(const Vec& u, const Vec& v) const {
        return u.equals(v);
    }

    /**
     * Vec Output stream
    */
    ostream& operator<<(ostream& os, const Vec& vec) {
        os << "(";
        for (int i=0; i<vec.vec.size(); i++) {
            os << vec.vec[i];
            if (i != vec.vec.size()-1) {
                os << ",";
            }
        }
        os << ")";
        return os;
    }



    /**
     * Vec operations with scalars
     */

    Vec operator+(const Vec& v, double s) {
        return Vec(v.vec + s);
    }
    Vec operator+(double s, const Vec& v) {
        return Vec(v.vec + s);
    }

    Vec operator-(const Vec& v, double s) {
        return Vec(v.vec - s);
    }
    Vec operator-(double s, const Vec& v) {
        return Vec(-v.vec + s);
    }

    Vec operator*(const Vec& v, double s) {
        return Vec(v.vec * s);
    }
    Vec operator*(double s, const Vec& v) {
        return Vec(v.vec * s);
    }
    
    Vec operator/(const Vec& v, const double s) {
        return Vec(v.vec / s);
    }
    Vec operator/(const double s, const Vec& v) {
        return Vec(s / v.vec);
    }


}