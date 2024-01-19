#include "mo/ball_list.h"

#include "mo/mo_helper.h"

#include <cmath>
#include <sstream>


using namespace std;

const double E = exp(1.0);
static double EPS = 1e-12;


namespace thts {
    

    /**
     * 
     * 
     * NGV
     * 
     * 
     * 
    */

    size_t NGV::hash() const 
    {
        return hash<Eigen::ArrayXd>()(point);
    };

    bool NGV::equals(const NGV& other) const 
    {
        return (point == other.point).all();
    };

    bool NGV::operator==(const NGV& other) const 
    {
        return equals(other);
    };

    bool NGV::operator!=(const NGV& other) const 
    {
        return !equals(other);
    };
    
    void NGV::share_values_message_passing() 
    {
        for (shared_ptr<NGV> other_ptr : *edges) {
            share_values_message_passing_helper(*other_ptr);
        }
    }

    void NGV::share_values_message_passing_helper(NGV& other) 
    {
        Eigen::ArrayXd ve_this = value_estimate;
        Eigen::ArrayXd ve_other = other.value_estimate;
        if (thts::helper::dot(point,ve_other) > thts::helper::dot(point,ve_this)) {
            value_estimate = ve_other;
        }
        if (thts::helper::dot(other.point,ve_this) > thts::helper::dot(other.point,ve_other)) {
            other.value_estimate = ve_this;
        }
    }

    NGV::NGV(Eigen::ArrayXd point, Eigen::ArrayXd init_val_estimate) : 
        value_esimtate(init_val_estimate), 
        point(point),
        edges(make_shared<unordered_set<shared_ptr<NGV>>>())
    {
    };

    /**
     * 
     * 
     * TN
     * 
     * 
     * 
    */

    bool TN::has_children() 
    {  
        return children->size() > 0;
    };

    size_t TN::hash() const 
    {
        return hash<Eigen::ArrayXd>()(point);
    };

    bool TN::equals(const NGV& other) const 
    {
        return (point == other.point).all();
    };

    bool TN::operator==(const NGV& other) const 
    {
        return equals(other);
    };

    bool TN::operator!=(const NGV& other) const 
    {
        return !equals(other);
    };

    /**
     * Getting value from this TN using simplex neighbourhood
    */
    Eigen::ArrayXd TN::get_value_estimate(const Eigen::ArrayXd& ctx) const 
    {
        double max_ctx_val = numeric_limits<double>::lowest();
        Eigen::ArrayXd max_val;
        for (shared_ptr<NGV> ngv_ptr : *simplex_neighbourhood) {
            double ctx_val = thts::helper::dot(ctx, ngv_ptr->value_estimate);
            if (ctx_val > max_ctx_val) {
                max_ctx_val = ctx_val;
                max_val = ngv_ptr->value_estimate;
            }
        }
        return max_val;
    }

    shared_ptr<TN> TN::traverse_one_step(const Eigen::ArrayXd& ctx) 
    {
        for (shared_ptr<TN> child_ptr : children) {
            if (other_tn_closer_to_ctx(ctx, *child_ptr)) {
                return child_ptr;
            }
        }
        return psuedo_child;
    }

    bool TN::other_tn_closer_to_ctx(Eigen::ArrayXd& ctx, TN& other) 
    {
        return thts::helper::dist(other.point, ctx) < thts::helper::dist(point, ctx);
    }

    TN::TN(int depth, Eigen::ArrayXd point) :
        depth(depth),
        point(point),
        children(make_shared<unordered_set<shared_ptr<TN>>>()),
        simplex_neighbourhood(make_shared<unordered_set<shared_ptr<NGV>>>()),
        lock()
    {   
    };


    /**
     * 
     * 
     * SimplexMap
     * 
     * 
     * 
    */

    SimplexMap::SimplexMap(int reward_dim, Eigen::ArrayXd default_val) :
        default_val(default_val),
        root_node(),
        tree_nodes(make_shared<vector<shared_ptr<TN>>>()),
        n_graph_vertices(make_shared<unordered_set<shared_ptr<NGV>>>()),
        // n_graph_edges(),
        lock()
    {
        double weight = 1.0 / reward_dim;
        Eigen::ArrayXd centroid = Eigen::ArrayXd::Constant(dim, weight);
        root_node = make_shared<TN>(0,centroid);
        tree_nodes->push_back(root_node);

        for (int i=0; i<reward_dim; i++) {
            Eigen::ArrayXd unit_vec = Eigen::ArrayXd::Zero(dim);
            unit_vec[i] = 1.0;
            n_graph_vertex = make_shared<NGV>(unit_vec,default_val);
            n_graph_vertices->insert(n_graph_vertex);
            root_node->simplex_neighbourhood.insert(n_graph_vertex);
        }

        for (shared_ptr<NGV> v1 : *n_graph_vertices) {
            for (shared_ptr<NGV> v2 : *n_graph_vertices) {
                if (!v1->equals(*v2)) {
                    v1->edges.insert(v2);
                    v2->edges.insert(v1);
                }
            }
        }
    }

    /**
     * Lookup closest tree node
    */
    shared_ptr<TN> SimplexMap::operator[](const Eigen::ArrayXd& ctx) const
    {
        return lookup_node_closest_to_context(ctx);
    }

    shared_ptr<TN> SimplexMap::lookup_node_closest_to_context(const Eigen::ArrayXd& ctx) const
    {
        lock_guard<mutex> lg(lock);
        shared_ptr<TN> cur_tn = root_node;
        while (cur_tn.has_children()) {
            cur_tn = cur_tn->traverse_one_step(ctx);
        }
        return cur_tn;
    }

    /**
     * Splitting
    */
    void SimplexMap::split_at(shared_ptr<TN> tn_ptr) {
        lock_guard<mutex> lg(lock);
        if (tn_ptr->has_children()) {
            throw runtime_error("Trying to split a simplex map tree node that already has children");
        }
        shared_ptr<unordered_set<shared_ptr<NGV>>> old_simplex_neighbourhood = tn_ptr->simplex_neighbourhood;

        // compute new simplex_neighbourhood points for tn_ptr, and add those NGV's to graph
        // For the smaaller simplex that will be formed of completely new NGVs and at the center
        shared_ptr<unordered_set<shared_ptr<NGV>>> new_central_simplex_neighbourhood =
            make_shared<unordered_set<shared_ptr<NGV>>>();
        unordered_map<shared_ptr<NGV>,shared_ptr<NGV>> cached_opposing_old_ngv_to_new_ngv(
            old_simplex_neighbourhood->size());
        for (shared_ptr<NGV> opposing_simplex_vertex : *old_simplex_neighbourhood) 
        {
            // make point at centroid of {old_simplex_neighbourhood} - {opposing simplex point}
            Eigen::ArrayXd opposing_centroid = Eigen::ArrayXd::Zero(default_val.size());
            for (shared_ptr<NGV> old_simplex_vertex : *old_simplex_neighbourhood) {
                if (old_simplex_vertex->equals(*opposing_simplex_vertex)) {
                    continue;
                }
                opposing_centroid += old_simplex_vertex->point;
            }
            opposing_centroid /= (old_simplex_neighbourhood.size() - 1.0);
            shared_ptr<NGV> opposing_ngv = make_shared<NGV>(opposing_centroid,default_val);

            // add edges to neighbourhood graph
            for (shared_ptr<NGV> old_simplex_vertex : *old_simplex_neighbourhood) {
                if (old_simplex_vertex->equals(*opposing_simplex_vertex)) {
                    continue;
                }
                old_simplex_vertex->edges->insert(opposing_ngv);
                opposing_ngv->edges->insert(old_simplex_vertex);
            }

            // Add ngv to datastruct
            n_graph_vertices->insert(opposing_ngv);
            // Add to new simplex neighbourhood points
            new_central_simplex_neighbourhood->insert(opposing_ngv);
            // NB: opposing_ngv is on the face opposite opposing_simplex_vertex
            cached_opposing_old_ngv_to_new_ngv[opposing_simplex_vertex] = opposing_ngv;
        }

        // Update simplex neighbourhood of tn_ptr
        tn_ptr->simplex_neighbourhood = new_central_simplex_neighbourhood;

        // Make child nodes
        for (shared_ptr<NGV> old_vertex : *old_simplex_neighbourhood) {
            // compute simplex of the new TN first
            shared_ptr<unordered_set<shared_ptr<NGV>>> child_simplex = make_shared<unordered_set<shared_ptr<NGV>>>(
                old_simplex_neighbourhood->size());
            child_simplex->insert(old_vertex);
            shared_ptr<NGV> ngv_to_ignore = cached_opposing_old_ngv_to_new_ngv[old_vertex];
            for (shared_ptr<NGV> central_simplex_vertex : *new_central_simplex_neighbourhood) {
                if (ngv_to_ignore.equals(central_simplex_vertex)) {
                    continue;
                }
                child_simplex->insert(central_simplex_vertex);
            }

            // Compute TN point as centroid of the simplex
            Eigen::ArrayXd new_tn_point = Eigen::ArrayXd::Zero(default_val.size());
            for (shared_ptr<NGV> child_simplex_vertex : *child_simplex) {
                new_tn_point += child_simplex_vertex->point;
            }
            new_tn_point /= (double) (child_simplex.size());

            // Create new node
            shared_ptr<TN> new_child = make_shared<TN>(tn_ptr->depth+1,new_tn_point);
            new_child->simplex_neighbourhood = child_simplex;

            // Add links
            tree_nodes->push_back(new_child);
            tn_ptr->children->insert(new_child);
        }

        // Remember to make psuedo child
        // DONT add it to 'tree_nodes', because using that to sample a 'true point' uniformly from the map
        tn_ptr->psuedo_child = make_shared<TN>(tn_ptr->depth+1,tn_ptr->point);
        th_ptr->psuedo_child->simplex_neighbourhood = new_central_simplex_neighbourhood;
        unordered_map<shared_ptr<NGV>,shared_ptr<unordered_set<shared_ptr<NGV>>> cached_sub_simplex(
            old_simplex_neighbourhood->size());
    }

    /**
     * Sample a random node
    */
    std::shared_ptr<TN> get_random_node(RandManager& rand_manager) 
    {
        lock_guard<mutex> lg(lock);
        int indx = rand_manager.get_rand_int(0, tree_nodes->size());
        return tree_nodes->at(indx);
    }

    /**
     * Prett print
    */
    std::string get_pretty_print_string()
    {
        stringstream ss;
        ss << "Simplex map pretty print: {" << endl;
        ss << "Weight // Value" << endl
        lock_guard<mutex> lg(lock);
        for (shared_ptr<NVG> v : *n_graph_vertices) {
            ss << "[";
            for (int i=0; i<v->point.size(); i++) {
                ss << v->point[i] << ",";
            }
            ss << "] // [";
            for (int i=0; i<v->value_estimate.size(); i++) {
                ss << v->value_estimate[i] << ",";
            }
            ss << "]" << endl;
        }
        ss << "}" << endl;
        return ss.str();
    }
}







/**
 * haash
*/
namespace std {
    size_t hash<shared_ptr<NGV>>::operator()(const shared_ptr<NGV>& v) const {
        return v->hash();
    }

    size_t hash<shared_ptr<TN>>::operator()(const shared_ptr<TN>& n) const {
        return v->hash();
    }
}