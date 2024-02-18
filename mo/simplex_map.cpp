#include "mo/simplex_map.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <utility>

#include <Eigen/SVD>


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

    NGV::NGV(Eigen::ArrayXd weight, Eigen::ArrayXd init_val_estimate, double init_entr_estimate) : 
        value_estimate(init_val_estimate), 
        entropy(init_entr_estimate),
        weight(weight),
        neighbours(make_shared<unordered_set<shared_ptr<NGV>>>())
    {
    };

    NGV::NGV(NGV& v0, NGV& v1, double ratio) :
        value_estimate(),
        entropy(),
        weight(),
        neighbours(make_shared<unordered_set<shared_ptr<NGV>>>())
    {
        // Weight interpolated
        weight = ratio * v0.weight + (1.0-ratio) * v1.weight;

        // Take the best value estimate from the two endpoints
        double ctx_val_v0 = thts::helper::dot(weight,v0.value_estimate);
        double ctx_val_v1 = thts::helper::dot(weight,v1.value_estimate);
        if (ctx_val_v0 >= ctx_val_v1) {
            value_estimate = v0.value_estimate;
            entropy = v0.entropy;
        } else {
            value_estimate = v1.value_estimate;
            entropy = v1.entropy;
        }

        // NOTE: we can't upsed the neighbourhood graph here because we cant call 'shared_from_this' from constructor
        // Also this should be done from LSE::insert anyway, because we dont necessarily know that v0 and v1 are still 
        // connected (i.e. there could be another point on the LSE between v0 and v1)
    };


    size_t NGV::hash() const 
    {
        return std::hash<Eigen::ArrayXd>()(weight);
    };

    bool NGV::equals(const NGV& other) const 
    {
        return (weight == other.weight).all();
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
        share_values_message_passing_push();
        share_values_message_passing_pull();
    }
    
    void NGV::share_values_message_passing_push() 
    {
        for (shared_ptr<NGV> other_ptr : *neighbours) {
            share_values_message_passing_helper_push(*other_ptr);
        }
    }
    
    void NGV::share_values_message_passing_pull() 
    {
        for (shared_ptr<NGV> other_ptr : *neighbours) {
            share_values_message_passing_helper_pull(*other_ptr);
        }
    }

    void NGV::share_values_message_passing_helper_push(NGV& other) 
    {
        if (thts::helper::dot(other.weight,value_estimate) > thts::helper::dot(other.weight,other.value_estimate)) {
            other.value_estimate = value_estimate;
            other.entropy = entropy;
        }
    }

    void NGV::share_values_message_passing_helper_pull(NGV& other) 
    {
        if (thts::helper::dot(weight,other.value_estimate) > thts::helper::dot(weight,value_estimate)) {
            value_estimate = other.value_estimate;
            entropy = other.entropy;
        }
    }

    void NGV::add_connection(shared_ptr<NGV> other)
    {
        neighbours->insert(other);
        other->neighbours->insert(shared_from_this());
    }

    void NGV::erase_connection(shared_ptr<NGV> other)
    {
        neighbours->erase(other);
        other->neighbours->erase(shared_from_this());
    }

    double NGV::contextual_value_estimate() 
    {
        return thts::helper::dot(weight, value_estimate);
    }

    double NGV::contextual_value_estimate(Eigen::ArrayXd& ctx) 
    {
        return thts::helper::dot(ctx, value_estimate);
    }
    
    /**
     * 
     * 
     * LSE
     * 
     * 
     * 
    */

    LSE::LSE(shared_ptr<NGV> v0, shared_ptr<NGV> v1) : 
        v0(v0), 
        v1(v1), 
        ratios(), 
        interpolated_vertex_tree(make_shared<LSE_BT>())
    {
        ratios[v0] = 0.0;
        ratios[v1] = 1.0;
    };
    
    size_t LSE::hash() const
    {
        return thts::helper::unordered_hash(v0,v1);
    };

    bool LSE::equals(const LSE& other) const 
    {
        return ((v0->equals(*other.v0) && v0->equals(*other.v0))
                || (v0->equals(*other.v1) && v1->equals(*other.v0)));
    };

    bool LSE::operator==(const LSE& other) const 
    {
        return equals(other);
    };

    bool LSE::operator!=(const LSE& other) const 
    {
        return !equals(other);
    };

    /**
     * TODO: be more efficient in not creating nodes before actually using them
     * 
     * If we call this function, then we are probably creating children TN's.
     * In such case, we will be making simplices with edges (x0,v) and (v,x1)
     * We want this edge to be returned when looking up the LSE for the (x0,v) and (v,x1) unordered pairs
     * 
     * We need to update the neighbourhood graph to account for the new vertex on this edge
     * (note that there may be other NGV's between x0 and x1)
     * This is what we use the binary tree for (to efficiently keep the vertices in order along this edge and to 
     * efficiently find the left and right neightbours of the new vertex along the edge)
    */
    void LSE::insert(shared_ptr<NGV> v, shared_ptr<NGV> x0, shared_ptr<NGV> x1, double r, SimplexMap& simplex_map) 
    {
        double x0_ratio = ratios.at(x0);
        double x1_ratio = ratios.at(x1);
        
        // Assert x0_ratio < x1_ratio
        if (x0_ratio > x1_ratio) {
            return insert(v, x1, x0, 1.0-r, simplex_map);
        }

        // ratio between v0 and v1 (endpoints of the LSE)
        double v_ratio = x0_ratio + (x1_ratio-x0_ratio) * r;

        // Find the LSE_BT node to insert v in
        // Also find the NGV that will be to the left and right of v
        shared_ptr<LSE_BT> bt_node = interpolated_vertex_tree;
        shared_ptr<NGV> left_vertex = v0;
        shared_ptr<NGV> right_vertex = v1;
        while (bt_node->vertex != nullptr) {
            if (v_ratio < bt_node->ratio) {
                right_vertex = bt_node->vertex;
                bt_node = bt_node->left;
            } else { // if (v_ratio > bt_node->ratio)
                left_vertex = bt_node->vertex;
                bt_node = bt_node->right;
            }
        }

        // Insert v in this LSE
        ratios[v] = v_ratio;
        bt_node->vertex = v;
        bt_node->ratio = v_ratio;
        bt_node->left = make_shared<LSE_BT>();
        bt_node->right = make_shared<LSE_BT>();

        // Update the neighbourhood graph
        left_vertex->erase_connection(right_vertex);
        left_vertex->add_connection(v);
        right_vertex->add_connection(v);

        // Register the new pairs of vertices with this LSE
        shared_ptr<LSE> this_lse = shared_from_this();
        simplex_map.register_vertices_with_lse(x0, v, this_lse);
        simplex_map.register_vertices_with_lse(v, x1, this_lse);
    }
    
    /**
     * 
     * 
     * Triangulation
     * 
     * 
     * 
    */

    vector<string> split_string(string s, string delimiter) {
        vector<string> vec;;
        size_t last = 0; 
        size_t next = 0; 
        while ((next = s.find(delimiter, last)) != string::npos) { 
            vec.push_back(s.substr(last, next-last));
            last = next+1;
        } 
        vec.push_back(s.substr(last));
        return vec;
    }

    /**
     * TODO: do some more robust file stuff? Feels a bit off having hard coded dir from root dir
    */
    Triangulation::Triangulation(int dim) : d(dim), e(dim * (dim-1) / 2), edge_points(), simplices()
    {
        // If passed zero, then not using triangulation
        if (dim == 0) {
            return;
        }

        // Filename for this triangulation
        stringstream ss;
        ss << "mo/.cache/" << dim << "_triangulation.txt";
        ifstream file(ss.str());
        if (!file.is_open()) {
            throw runtime_error("Error opening precomputed triangulation text file");
        }

        // read num vertices line
        string line;
        getline(file, line);
        int num_vertices = stoi(line);
        if (num_vertices != d+e) {
            throw runtime_error("Unexpected number of vertices in the triangulation file");
        }

        // read num simplices line
        getline(file, line);
        int num_simplices = stoi(line);

        // skip through d lines that will read 0\n1\n...(d-1)\n
        for (int i=0; i<d; i++) {
            getline(file, line);
        }

        // read in the e lines that define the edge points
        for (int i=0; i<e; i++) {
            getline(file, line);
            vector<string> edge_point_info = split_string(line, " ");
            int index0 = stoi(edge_point_info[1]);
            int index1 = stoi(edge_point_info[2]);
            double ratio = stod(edge_point_info[3]);
            edge_points.push_back(tuple<int,int,double>(index0, index1, ratio));
        }

        // read in the lists of vertices that form the simplices
        for (int i=0; i<num_simplices; i++) {
            getline(file, line);
            vector<string> simplex_indices = split_string(line, " ");
            simplices.push_back(vector<int>());
            for (string& index_str : simplex_indices) {
                simplices[i].push_back(stoi(index_str));
            }
        }

        // close file as finished
        file.close();
    };


    /**
     * 
     * 
     * TN
     * 
     * 
     * 
    */
    TN::TN(int dim, int depth, shared_ptr<vector<shared_ptr<NGV>>> simplex_vertices) :
        dim(dim),
        depth(depth),
        centroid(Eigen::ArrayXd::Zero(dim)),
        l_inf_norm(0.0),
        split_counter(0),
        simplex_vertices(simplex_vertices),
        hyperplane_normals(make_shared<unordered_map<shared_ptr<NGV>,Eigen::ArrayXd>>()),
        children(make_shared<unordered_set<shared_ptr<TN>>>()),
        splitting_edge_normal_side_vertex(),
        splitting_edge_opposite_side_vertex(),
        splitting_edge_new_vertex(),
        splitting_hyperplane_normal(Eigen::ArrayXd::Zero(dim)),
        normal_side_child(),
        opposite_side_child()
    {
        // Compute centroid
        for (shared_ptr<NGV> vertex : *simplex_vertices) {
            centroid += vertex->weight;
        }
        centroid /= simplex_vertices->size();

        // Compute l_inf_norm
        for (size_t i=0; i<simplex_vertices->size(); i++) {
            for (size_t j=i+1; j<simplex_vertices->size(); j++) {
                Eigen::ArrayXd diff = simplex_vertices->at(i)->weight - simplex_vertices->at(j)->weight;
                double diff_l_inf_norm = diff.abs().maxCoeff();
                if (diff_l_inf_norm > l_inf_norm) {
                    l_inf_norm = diff_l_inf_norm;
                    splitting_edge_normal_side_vertex = simplex_vertices->at(i);
                    splitting_edge_opposite_side_vertex = simplex_vertices->at(j);
                }
            }
        }

        // Ensure neighbourhood graph is actually connected=
        _ensure_neighbourhood_graph_connected();
    }

    void TN::_ensure_neighbourhood_graph_connected() 
    {
        for (size_t i=0; i<simplex_vertices->size(); i++) {
            for (size_t j=i+1; j<simplex_vertices->size(); j++) {
                simplex_vertices->at(i)->add_connection(simplex_vertices->at(j));
            }
        }
    }

    /**
     * TODO: move relevant docstring from 'lazy_compute_hyperplane_noamrls' to here
     * 
     * hyperplane_points contains 'dim-1' many points defining a 'dim-2' hyperplane
    */
    Eigen::ArrayXd TN::compute_hyperplane_normal(vector<shared_ptr<NGV>>& hyperplane_points) const
    {
        // Construct the (D,D-1) matrix we want to SVD
        // Fill the first collumn with 1's (as described in above comment)
        // Fill remaining collumns with the D-2 values of opposing_face_vertices[i] - oppositing_face_vertices[0]
        Eigen::MatrixXd hyperplane_matrix(dim,dim-1);
        hyperplane_matrix.col(0).setOnes();
        hyperplane_matrix.col(0) /= dim;
        if (hyperplane_points.size() > 1) {
            Eigen::VectorXd v_0 = hyperplane_points[0]->weight.matrix();
            for (size_t i=1; i<hyperplane_points.size(); i++) {
                Eigen::VectorXd v_i = hyperplane_points[i]->weight.matrix();
                hyperplane_matrix.col(i) = v_i - v_0;
            }
        }

        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeThinV> svd(hyperplane_matrix);

        // If SVD is M=USV^T, then we want U.col(d-1), so read that out
        // Note that S(i,i) >= S(i+1,i+1), as singular values computed in order from largest to smallest
        // Also convert back to ArrayXd type, done doing lin alg stuff
        return svd.matrixU().col(dim-1).array();
    }

    /**
     * This is a bit complex, so I'll write some comments about this
     * We are working in D dimensions (with D rewards)
     * That means we're using a D-1 simplex (with D points)
     * This D-1 simplex lies in a D-1 dimensional hyperplane of the D dimensional plane
     * (1,1,1,...,1) is the normal to this D-1 dimensional hyperplane
     * The D-1 simplex has D many D-2 faces (which defines a D-2 hyperplane)
     * 
     * For each v in simplex_vertices, hyperplane_normals[v] is normal to the D-2 face opposing v
     * (For example in 2dimensions (with 3 rewards) the hyperplane is the line opposite the point) 
     * 
     * For each normal we compute, we compute the normal to D-1 hyperplane that coincides with the D-2 face of the 
     * simplex. The additional dimension of this D-1 hyperplane extends out from the D-1 simplex
     * - to visualise this, consider the triangle (2-simplex) in 3d space, and we're computing the planes that 
     *      intersect the lines of the triangle
     * 
     * Now, suppose we have k points v1,...,vk that lie on a k-1 hyperplane in kd space, how do we compute the normal?
     * As v1 + c * (vi - v1) lies in the plane, we have the plane extending in the direction (vi-v1)
     * So consider the matrix M with collumn vectors ((v2-v1) (v3-v1) ... (vk-v1)), which is a (k,k-1) matrix
     * The normal vector to the plane is the null space of this matrix
     * So we can compute the SVD of M, and consider the vector corresponding to the singular (eigen) value of zero
     * 
     * Returning to the problem at hand, we get the points v1,...,v(D-1) from simplex_vertices - {v} 
     * (recall v is the simplex vertex that opposes the face we are currently considering)
     * These D-1 points define the D-2 hyperplane we want
     * We let vD = v1 + 1, so get D points defining a D-1 hyperplane
     * And note that if vD = v1 + 1, then vD - v1 = 1
     * 
     * NOTE: this could probably be implemented a bit more efficiently by actually projecting into the D-1 space and 
     *  working directly in that dimension. But the above is how my brain thought about it, and I just want something 
     *  that works for now.
    */
    void TN::lazy_compute_hyperplane_normals() const 
    {
        // If already computed or working in 2d, do nothing
        if (dim == 2 || hyperplane_normals->size() > 0) {
            return;
        }

        for (shared_ptr<NGV> opposing_vertex : *simplex_vertices) {
            // Get D-1 opposing face vertices
            vector<shared_ptr<NGV>> opposing_face_vertices;
            for (shared_ptr<NGV> vertex : *simplex_vertices) {
                if (vertex == opposing_vertex) {
                    continue;
                }
                opposing_face_vertices.push_back(vertex);
            }

            // Compute hyperplane normal
            Eigen::ArrayXd normal = compute_hyperplane_normal(opposing_face_vertices);

            // Make sure that normal points towards centroid
            if (thts::helper::dot(centroid - opposing_face_vertices[0]->weight, normal) < 0.0) {
                normal *= -1.0;
            }

            // Insert
            hyperplane_normals->insert_or_assign(opposing_vertex, normal);
        }
    }

    // size_t TN::hash() const 
    // {
    //     return hash<Eigen::ArrayXd>()(centroid);
    // };

    // bool TN::equals(const NGV& other) const 
    // {
    //     return (centroid == other.centroid).all();
    // };

    // bool TN::operator==(const NGV& other) const 
    // {
    //     return equals(other);
    // };

    // bool TN::operator!=(const NGV& other) const 
    // {
    //     return !equals(other);
    // };

    bool TN::has_children() const 
    {  
        return children->size() > 0 || normal_side_child != nullptr;
    }

    void TN::create_children(SimplexMap& simplex_map, SmtThtsManager& sm_manager) 
    {
        if (sm_manager.use_triangulation) {
            return create_children_triangulation(simplex_map, sm_manager);
        } else {
            return create_children_binary_tree(simplex_map, sm_manager);
        }
    }
    
    /**
     * TODO: document a bit better generally
     * TODO: 
    */
    void TN::create_children_binary_tree(SimplexMap& simplex_map, SmtThtsManager& sm_manager) 
    {
        // create the new vertex on the splitting edge (halfway between the two)
        // avoid making a duplicate vertex, and add it to simplex map structures if made a novel vertex
        splitting_edge_new_vertex = make_shared<NGV>(
            *splitting_edge_normal_side_vertex, *splitting_edge_opposite_side_vertex, 0.5);
        if (simplex_map.n_graph_vertex_set->contains(splitting_edge_new_vertex)) {
            splitting_edge_new_vertex = *simplex_map.n_graph_vertex_set->find(splitting_edge_new_vertex);
        } else {
            simplex_map.n_graph_vertices->push_back(splitting_edge_new_vertex);
            simplex_map.n_graph_vertex_set->insert(splitting_edge_new_vertex);
        }

        // Insert it on the LSE
        shared_ptr<LSE> simplex_edge = simplex_map.get_or_create_lse(
            splitting_edge_normal_side_vertex, splitting_edge_opposite_side_vertex);
        simplex_edge->insert(
            splitting_edge_new_vertex, 
            splitting_edge_normal_side_vertex, 
            splitting_edge_opposite_side_vertex,
            0.5,
            simplex_map);
        
        // Create vector of all vertices common to both children
        vector<shared_ptr<NGV>> child_common_simplex_vertices;
        for (shared_ptr<NGV> simplex_vertex : *simplex_vertices) {
            if ((*simplex_vertex != *splitting_edge_normal_side_vertex) 
                && (*simplex_vertex != *splitting_edge_opposite_side_vertex))
            {
                child_common_simplex_vertices.push_back(simplex_vertex);
            }
        }
        child_common_simplex_vertices.push_back(splitting_edge_new_vertex);

        // Normal side child simplex
        shared_ptr<vector<shared_ptr<NGV>>> normal_side_child_vertices = make_shared<vector<shared_ptr<NGV>>>(
            child_common_simplex_vertices);
        normal_side_child_vertices->push_back(splitting_edge_normal_side_vertex);
        normal_side_child = make_shared<TN>(dim, depth+1, normal_side_child_vertices);

        // Opposite side child simplex
        shared_ptr<vector<shared_ptr<NGV>>> opposite_side_child_vertices = make_shared<vector<shared_ptr<NGV>>>(
            child_common_simplex_vertices);
        opposite_side_child_vertices->push_back(splitting_edge_opposite_side_vertex);
        opposite_side_child = make_shared<TN>(dim, depth+1, opposite_side_child_vertices);

        // Compute normal (using the dim-1 many common points of the child simplices)
        splitting_hyperplane_normal = compute_hyperplane_normal(child_common_simplex_vertices);

        // and make sure that the normal points towards the normal side child
        Eigen::ArrayXd splitting_edge_normal_dir = (splitting_edge_normal_side_vertex->weight 
                                                    - splitting_edge_new_vertex->weight);
        if (thts::helper::dot(splitting_edge_normal_dir, splitting_hyperplane_normal) < 0.0) {
            splitting_hyperplane_normal *= -1.0;
        }
    }

    /**
     * We perform the following steps:
     * 0. In the following we will have D+E points that we triangulate over
     * 0.1. The first 0,...,D-1 vertices are copies from 'simplex_vertices'
     * 0.2. The next D,...,D+E-1 vertices are created along the edges of the simplex
     * 1. For each LSE of the simplex, we add a new NGV along it
     * 1.1. This NGV needs to be inserted into the LSE
     * 1.2. Note that LSE.insert will update the neightbourhood graph and register the lse with the new 'subedges' 
     *      being created
     * 2. For each simplex (list of vertices) make a new TN
     * 2.1. The TN constructor will ensure that the vertices are connected
     * 
     * Note that we need to take care to not make any duplicate vertices in the neighbourhood graph
    */
    void TN::create_children_triangulation(SimplexMap& simplex_map, SmtThtsManager& sm_manager) 
    {
        // 0+1: make the list of vertices to triangulate over
        vector<shared_ptr<NGV>> vertices(*simplex_vertices);
        for (tuple<int,int,double>& edge_point_spec : sm_manager.triangulation_ptr->edge_points) {
            shared_ptr<NGV> v0 = vertices[get<0>(edge_point_spec)];
            shared_ptr<NGV> v1 = vertices[get<1>(edge_point_spec)];
            double ratio = get<2>(edge_point_spec);

            shared_ptr<NGV> new_vertex = make_shared<NGV>(*v0, *v1, ratio);
            if (simplex_map.n_graph_vertex_set->contains(new_vertex)) {
                new_vertex = *simplex_map.n_graph_vertex_set->find(new_vertex);
            } else {
                simplex_map.n_graph_vertices->push_back(new_vertex);
                simplex_map.n_graph_vertex_set->insert(new_vertex);
            }
            vertices.push_back(new_vertex);

            shared_ptr<LSE> simplex_edge = simplex_map.get_or_create_lse(v0,v1);
            simplex_edge->insert(new_vertex, v0, v1, ratio, simplex_map);
        }

        // 2: make child simplices
        for (vector<int>& simplex_indices : sm_manager.triangulation_ptr->simplices) {
            shared_ptr<vector<shared_ptr<NGV>>> child_simplex_vertices = make_shared<vector<shared_ptr<NGV>>>();
            for (int& i : simplex_indices) {
                child_simplex_vertices->push_back(vertices[i]);
            }
            shared_ptr<TN> child_tn = make_shared<TN>(dim, depth+1, child_simplex_vertices);
            children->insert(child_tn);
        }
    }

    /**
     * We can tell if this TN node is a binary tree or used triangulation depending on if any of the following 
     * pointers are nullptr or not:
     * - splitting_edge_new_vertex
     * - normal_side_child
     * - opposite_side_child
    */
    shared_ptr<TN> TN::get_child(const Eigen::ArrayXd& weight) const
    {
        if (normal_side_child == nullptr) {
            return get_child_triangulation(weight);
        } else {
            return get_child_binary_tree(weight);
        }
    }

    shared_ptr<TN> TN::get_child_binary_tree(const Eigen::ArrayXd& weight) const 
    {
        // if (dim == 2) {
        //     if (normal_side_child->contains_weight_2d(weight)) {
        //         return normal_side_child;
        //     } else {
        //         return opposite_side_child;
        //     }
        // }

        if (halfplane_check(splitting_edge_new_vertex->weight, splitting_hyperplane_normal, weight)) {
            return normal_side_child;
        } else {
            return opposite_side_child;
        }
    }

    shared_ptr<TN> TN::get_child_triangulation(const Eigen::ArrayXd& weight) const
    {  
        for (shared_ptr<TN> child : *children) {
            if (child->contains_weight(weight)) {
                return child;
            }
        }
        cout << weight << endl;
        throw runtime_error("Either called get child without children, or probably called with weight with vals not "
            "summing to one");
    }

    /**
     * Only called by triangulation version of the code at the moment
     * But chould generally 
    */
    bool TN::contains_weight(const Eigen::ArrayXd& weight, bool debug) const 
    {
        if (dim == 2) {
            return contains_weight_2d(weight);
        }

        // Ensure hyperplane normals are computed
        lazy_compute_hyperplane_normals();

        // Perform halfplane checks
        // if we fail any halfplane check, then 'weight' isnt in this simplex
        for (size_t i=0; i<simplex_vertices->size(); i++) {
            Eigen::ArrayXd halfplane_normal = hyperplane_normals->at(simplex_vertices->at(i));
            Eigen::ArrayXd halfplane_point = simplex_vertices->at(0)->weight;
            if (i==0) halfplane_point = simplex_vertices->at(1)->weight;
            if (!halfplane_check(halfplane_point, halfplane_normal, weight)) {
                return false;
            }
        }

        // If get here, passed all halfplane checks
        return true;
    }

    /**
     * For 2d case using hyperplanes and the same logic is a little overkill
     * Can just use the first dimension of the weight to check contains
     * As a 1d simplex is just a line
    */
    bool TN::contains_weight_2d(const Eigen::ArrayXd& weight) const 
    {
        shared_ptr<NGV> beg = simplex_vertices->at(0);
        shared_ptr<NGV> end = simplex_vertices->at(1);
        if (beg->weight[0] > end->weight[0]) {
            std::swap(beg,end);
        }
        return beg->weight[0] <= weight[0] && weight[0] <= end->weight[0];
    }

    /**
     * Helper for numerical instability
    */
    bool is_approx_zero(double x) {
        return -EPS < x && x < EPS;
    }
    
    /**
     * return true if point is in plane (i.e. if wegith-halfplane_point dot halfplane_normal == 0)
     * use approx == 0 for numerical errors
     * aprox== 0 will only be satisfied if plane is in point, shouldn't really be getting to simplices so small that 
     *      values below abs(EPS) are relevant
     * 
     * Note that we ensured while computing hyperplane normals that the normal points towards the centroid
    */
    bool TN::halfplane_check(
        const Eigen::ArrayXd& halfplane_point, 
        const Eigen::ArrayXd& halfplane_normal, 
        const Eigen::ArrayXd& weight) const 
    {
        double dot_prod = thts::helper::dot(weight-halfplane_point, halfplane_normal);
        return dot_prod >= 0 || is_approx_zero(dot_prod);
    }

    shared_ptr<NGV> TN::get_closest_ngv_vertex(const Eigen::ArrayXd& ctx) const
    {
        double closest_dist = std::numeric_limits<double>::max();
        shared_ptr<NGV> closest_vertex;
        for (shared_ptr<NGV> vertex : *simplex_vertices) {
            double l2_dist = thts::helper::dist(vertex->weight, ctx);
            if (l2_dist < closest_dist) {
                closest_dist = l2_dist;
                closest_vertex = vertex;
            }
        }
        if (closest_vertex == nullptr) {
            throw runtime_error("Error in indexing simplex map");
        }
        return closest_vertex;
    }

    shared_ptr<NGV> TN::operator[](const Eigen::ArrayXd& ctx) const
    {
        return get_closest_ngv_vertex(ctx);
    }

    /**
     * Getting value from this TN using simplex neighbourhood
    */
    Eigen::ArrayXd TN::get_best_value_estimate(const Eigen::ArrayXd& ctx) const 
    {
        double max_ctx_val = numeric_limits<double>::lowest();
        Eigen::ArrayXd max_val;
        for (shared_ptr<NGV> ngv_ptr : *simplex_vertices) {
            double ctx_val = thts::helper::dot(ctx, ngv_ptr->value_estimate);
            if (ctx_val > max_ctx_val) {
                max_ctx_val = ctx_val;
                max_val = ngv_ptr->value_estimate;
            }
        }
        return max_val;
    }

    void TN::maybe_subdivide(SimplexMap& simplex_map, SmtThtsManager& sm_manager)
    {
        // If already subdivided, no need
        if (has_children()) {
            return;
        }

        // If past a threshold, then we should forever remain a leaf node
        if (depth >= sm_manager.simplex_node_max_depth) {
            return;
        }
        if (l_inf_norm <= sm_manager.simplex_node_l_inf_thresh) {
            return;
        }

        // We might want to split if we get here
        // Check if any value estimates in our simplex are different
        bool non_uniform_value = false;
        Eigen::ArrayXd& ref_val_estimate = simplex_vertices->at(0)->value_estimate;
        for (size_t i=1; i<simplex_vertices->size(); i++) {
            if ((ref_val_estimate != simplex_vertices->at(i)->value_estimate).any()) {
                non_uniform_value = true;
                break;
            }
        }

        // update counter, any maybe split
        if (non_uniform_value) {
            split_counter++;
        } else {
            split_counter = 0;
        }

        if (split_counter >= sm_manager.simplex_node_split_visit_thresh) {
            create_children(simplex_map, sm_manager);
        }
    }


    /**
     * 
     * 
     * SimplexMap
     * 
     * 
     * 
    */

    SimplexMap::SimplexMap(int reward_dim, Eigen::ArrayXd default_val) :
        dim(reward_dim),
        root_node(),
        n_graph_vertices(make_shared<vector<shared_ptr<NGV>>>()),
        n_graph_vertex_set(make_shared<unordered_set<shared_ptr<NGV>>>()),
        lse_map()
    {
        // Make neighbourhood graph vertices for unit basis vectors (unit simplex)
        // Register in n_graph_vertices
        shared_ptr<vector<shared_ptr<NGV>>> unit_simplex_vertices = make_shared<vector<shared_ptr<NGV>>>();
        for (int i=0; i<dim; i++) {
            Eigen::ArrayXd basis_vector = Eigen::ArrayXd::Zero(dim);
            basis_vector[i] = 1.0;
            shared_ptr<NGV> simplex_vertex = make_shared<NGV>(basis_vector, default_val, 0.0);
            n_graph_vertices->push_back(simplex_vertex);
            n_graph_vertex_set->insert(simplex_vertex);
            unit_simplex_vertices->push_back(simplex_vertex);
        }    

        // Make root TN node with unit simplex
        root_node = make_shared<TN>(dim, 0, unit_simplex_vertices);
    }
    
    shared_ptr<LSE> SimplexMap::get_or_create_lse(shared_ptr<NGV> v0, shared_ptr<NGV> v1) 
    {
        UnorderedNGVPair lse_map_key = UnorderedNGVPair(v0,v1);
        if (!lse_map.contains(lse_map_key)) {
            lse_map[lse_map_key] = make_shared<LSE>(v0,v1);
        }
        return lse_map[lse_map_key];
    }
    
    void SimplexMap::register_vertices_with_lse(shared_ptr<NGV> v0, shared_ptr<NGV> v1, shared_ptr<LSE> edge) 
    {
        UnorderedNGVPair lse_map_key = UnorderedNGVPair(v0,v1);
        lse_map[lse_map_key] = edge;
    }

    shared_ptr<TN> SimplexMap::get_leaf_tn_node(const Eigen::ArrayXd& ctx) const 
    {
        shared_ptr<TN> cur = root_node;
        while (cur->has_children()) {
            cur = cur->get_child(ctx);
        }
        return cur;
    }
    
    shared_ptr<TN> SimplexMap::operator[](const Eigen::ArrayXd& ctx) const
    {
        return get_leaf_tn_node(ctx);
    }

    shared_ptr<NGV> SimplexMap::sample_random_ngv_vertex(RandManager& rand_manager) const
    {
        int rand_index = rand_manager.get_rand_int(0,n_graph_vertices->size());
        return n_graph_vertices->at(rand_index);
    }

    /**
     * Prett print
    */
    std::string SimplexMap::get_pretty_print_string() const
    {
        stringstream ss;
        ss << "Simplex map pretty print: {" << endl;
        ss << "Weight // Value" << endl;
        for (shared_ptr<NGV> v : *n_graph_vertices) {
            ss << "[";
            for (int i=0; i<v->weight.size(); i++) {
                ss << v->weight[i] << ",";
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
    using namespace thts;

    size_t hash<shared_ptr<NGV>>::operator()(const shared_ptr<NGV>& v) const 
    {
        return v->hash();
    }

    size_t equal_to<shared_ptr<NGV>>::operator()(const shared_ptr<NGV>& v0, const shared_ptr<NGV>& v1) const 
    {
        return v0->equals(*v1);
    }

    size_t hash<shared_ptr<LSE>>::operator()(const shared_ptr<LSE>& e) const 
    {
        return e->hash();
    }

    size_t equal_to<shared_ptr<LSE>>::operator()(const shared_ptr<LSE>& e0, const shared_ptr<LSE>& e1) const 
    {
        return e0->equals(*e1);
    }

    size_t hash<UnorderedNGVPair>::operator()(const UnorderedNGVPair& p) const
    {
        return thts::helper::unordered_hash(p.first,p.second);
    }

    size_t equal_to<UnorderedNGVPair>::operator()(const UnorderedNGVPair& p0, const UnorderedNGVPair& p1) const
    {
        return ((p0.first->equals(*p1.first) && p0.second->equals(*p1.second))
            || (p0.first->equals(*p1.second) && p0.second->equals(*p1.first)));
    }
}