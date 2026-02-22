#include "mo/data_structures/simplex_map.h"

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

#include <iostream>


using namespace std;

const double E = exp(1.0);
static double EPS = 1e-12;

// ------------------------------------------------------------
// SMVertex
// ------------------------------------------------------------
namespace thts {

    /**
    * Constructor
    */
    SMVertex::SMVertex(const Vec& weight, const Vec& heuristic_value_estimate, double entropy_estimate) : 
        weight(weight),
        num_updates(0),
        value_estimate(Vec(weight.size(), 0.0)),
        value_estimate_for_search(heuristic_value_estimate),
        entropy_estimate(entropy_estimate),
        neighbours(make_shared<unordered_set<shared_ptr<SMVertex>>>())
    {
    };

    /**
    * Constructor as midpoint of two other vertices
    */
    SMVertex::SMVertex(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1, double ratio) : 
        weight(ratio * v0.weight + (1.0-ratio) * v1.weight),
        num_updates(0),
        value_estimate(v0.value_estimate),
        value_estimate_for_search(v0.value_estimate_for_search),
        entropy_estimate(v0.entropy_estimate),
        neighbours(make_shared<unordered_set<shared_ptr<SMVertex>>>())
    {
        if (v1.value_estimate.dot(this->weight) > v0.value_estimate.dot(this->weight)) {
            this->value_estimate = v1.value_estimate;
            this->value_estimate_for_search = v1.value_estimate_for_search;
            this->entropy_estimate = v1.entropy_estimate;
        }

        // Cannot call shared_from_this() from constructor, so need to do this manually
        // this->neighbours->insert(v0);
        // this->neighbours->insert(v1);
        // v0->neighbours->insert(shared_from_this());
        // v1->neighbours->insert(shared_from_this());
    };

    /**
     * Hash / compare
     */
    size_t SMVertex::hash() const 
    {
        return std::hash<Vec>()(weight);
    };

    bool SMVertex::equals(const SMVertex& other) const 
    {
        return weight.equals(other.weight);
    };

    bool SMVertex::operator==(const SMVertex& other) const 
    {
        return equals(other);
    };

    bool SMVertex::operator!=(const NGV& other) const 
    {
        return !equals(other);
    };
    
    /**
    * Message passing (BFS)
    */
    void SMVertex::share_values_message_passing(int max_push_radius=1) 
    {  
        int current_push_radius = 0;
        queue<shared_ptr<SMVertex>> vertex_queue;
        queue<shared_ptr<SMVertex>> next_vertex_queue;
        unordered_set<shared_ptr<SMVertex>> visited_vertices;
        vertex_queue.push(shared_from_this());
        visited_vertices.insert(shared_from_this());

        while (!vertex_queue.empty() && current_push_radius < max_push_radius) {
            shared_ptr<SMVertex> current_vertex = vertex_queue.front();
            vertex_queue.pop();
            for (shared_ptr<SMVertex> neighbour_ptr : *current_vertex->neighbours) {
                if (visited_vertices.contains(neighbour_ptr)) continue;
                visited_vertices.insert(neighbour_ptr);
                bool success = share_values_message_passing_helper_push(*current_vertex, *neighbour_ptr);
                if (success && !visited_vertices.contains(neighbour_ptr)) {
                    next_vertex_queue.push(neighbour_ptr);
                }
            }
            if (vertex_queue.empty()) {
                current_push_radius++;
                vertex_queue = next_vertex_queue;
                next_vertex_queue.clear();
            }
        }
    }

    /**
    * Old docstring about why we need shareable_value parameter
    *
     * Just copying docstring here to highlight that we only try to push/pull backed up values, and avoid copying 
     * heuristic value estimates. (This lead to problems when I was developing, but cant remember at time of writing).
     * 
     * The need for value_esimtate_from_backup comes from not wanting to share heuristic values. Consider a case for 
     * example where the heuristic is the zero vector [0,0], and all of your rewards are negative. So the values are 
     * [-a -b], for some a,b >= 0. In this case, the message passing will keep the heuristic values always, rather than 
     * the more accurate dp estimates. So we mark if the value estimate is from a backup, so we can avoid pulling 
     * innacurate heuristic values.
     */
    bool SMVertex::share_values_message_passing_helper(SMVertex& from_vertex, SMVertex& to_vertex) 
    {  
        if (from_vertex.num_updates <= 0)
        {
            return false;
        }
        if (from_vertex.value_estimate.dot(to_vertex.weight) > to_vertex.value_estimate.dot(to_vertex.weight)) 
        {
            to_vertex.num_updates = from_vertex.num_updates;
            to_vertex.value_estimate = from_vertex.value_estimate;
            to_vertex.value_estimate_for_search = from_vertex.value_estimate_for_search;
            to_vertex.entropy_estimate = from_vertex.entropy_estimate;
            return true;
        }
        return false;
    }

    /**
    * Graph connections
     */
    void SMVertex::add_bidirectional_connection(shared_ptr<SMVertex> other)
    {
        this->neighbours->insert(other);
        other->neighbours->insert(shared_from_this());
    }

    void SMVertex::erase_bidirectional_connection(shared_ptr<SMVertex> other)
    {
        this->neighbours->erase(other);
        other->neighbours->erase(shared_from_this());
    }
}




// ------------------------------------------------------------
// SMSimplex
// ------------------------------------------------------------


namespace thts {

    SMSimplex::SMSimplex(int dim, std::vector<std::shared_ptr<SMVertex>>& vertices, int depth) :
        dim(dim),
        vertices(vertices),
        vertices_set(vertices),
        split_counter(0),
        longest_edge(std::make_pair(nullptr, nullptr)), // set in constructor
        split_vertex(nullptr), // initialised when splitting
        splitting_hyperplane_normal(nullptr), // initialised when splitting
        depth(depth),
        radius(0.0), // set in constructor
        normal_child(nullptr), // initialised when splitting
        opposite_child(nullptr), // initialised when splitting
        is_non_conforming(false) // assume conforming, SMMesh will update if we are non-conforming
    {
        // If 2D, then we want to make sure that the vertices are in the correct order in longest_edge
        if (is_2d()) 
        {
            Vec v0 = this->vertices[0]->weight;
            Vec v1 = this->vertices[1]->weight;
            if (v0[0] > v1[0]) 
            {
                this->longest_edge = std::make_pair(v1, v0);
            }
            else
            {
                this->longest_edge = std::make_pair(v0, v1);
            }
            this->radius = v0.dist(v1);
            return;
        }

        // Compute radius and longest edge
        for (size_t i=0; i<vertices.size(); i++) {
            for (size_t j=i+1; j<vertices.size(); j++) {
                Vec diff = vertices.at(i)->weight - vertices.at(j)->weight;
                double dist = diff.norm();
                if (dist > radius) {
                    this->radius = dist;
                    this->longest_edge = std::make_pair(vertices.at(i), vertices.at(j));
                }
            }
        }
    }

    SMSimplex::is_2d() const
    {
        return this->dim == 2;
    }

    bool SMSimplex::contains_vertex(shared_ptr<SMVertex> vertex) const
    {
        return this->vertices_set.contains(vertex);
    }

    /**
        Mashed together old docstrings from previous implementation

     * This is a bit complex, so I'll write some comments about this
     * We are working in D dimensions (with D rewards)
     * That means we're using a D-1 simplex (with D points)
     * This D-1 simplex lies in a D-1 dimensional hyperplane of the D dimensional plane
     * (1,1,1,...,1) is the normal to this D-1 dimensional hyperplane
     * 
     * Now, suppose we have k points v1,...,vk that lie on a k-1 hyperplane in kd space, how do we compute the normal?
     * As v1 + c * (vi - v1) lies in the plane, we have the plane extending in the direction (vi-v1)
     * So consider the matrix M with collumn vectors ((v2-v1) (v3-v1) ... (vk-v1)), which is a (k,k-1) matrix
     * The normal vector to the plane is the null space of this matrix
     * So we can compute the SVD of M, and consider the vector corresponding to the singular (eigen) value of zero
     * 
     We are assuming that the hyperplane_points are not colinear
     * 
     * NOTE: this could probably be implemented a bit more efficiently by actually projecting into the D-1 space and 
     *  working directly in that dimension. But the above is how my brain thought about it, and I just want something 
     *  that works for now.

     TO make use of eigen SVD, we will read out the underlying Eigen arrays to fill the matrix, and convert back to Vec 
     at the end
    */
    Vec SMSimplex::compute_hyperplane_normal(vector<shared_ptr<SMVertex>>& hyperplane_points) const
    {
        // Construct the (D,D-1) matrix we want to SVD
        // Fill the first collumn with 1's (as described in above comment)
        // Fill remaining collumns with the D-2 values of hyperplane_points[i] - hyperplane_points[0]
        Eigen::MatrixXd hyperplane_matrix(dim,dim-1);
        hyperplane_matrix.col(0).setOnes();
        hyperplane_matrix.col(0) /= dim;
        if (hyperplane_points.size() > 1) {
            Eigen::VectorXd v_0 = hyperplane_points[0]->weight.vec.matrix();
            for (size_t i=1; i<hyperplane_points.size(); i++) {
                Eigen::VectorXd v_i = hyperplane_points[i]->weight.vec.matrix();
                hyperplane_matrix.col(i) = v_i - v_0;
            }
        }

        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeThinV> svd(hyperplane_matrix);

        // If SVD is M=USV^T, then we want U.col(d-1), so read that out
        // Note that S(i,i) >= S(i+1,i+1), as singular values computed in order from largest to smallest
        // Also convert back to Vec type, done doing lin alg stuff
        return Vec(svd.matrixU().col(dim-1).array());
    }

    /**
     * Get the closest vertex to a weight from the points in this simplex
    */
    shared_ptr<SMVertex> SMSimplex::get_closest_vertex(const Vec& weight) const
    {
        double closest_dist = std::numeric_limits<double>::max();
        shared_ptr<SMVertex> closest_vertex;
        for (shared_ptr<SMVertex> vertex : vertices) {
            double dist = vertex->weight.dist(weight);
            if (dist < closest_dist) {
                closest_dist = dist;
                closest_vertex = vertex;
            }
        }
        return closest_vertex;
    }

    /**
     * Get the closest vertex to a weight from the points in this simplex
    */
    shared_ptr<SMVertex> SMSimplex::operator[](const Vec& weight) const
    {
        return this->get_closest_vertex(weight);
    }

    /**
     * See inline comments
    */
    void SMSimplex::create_children(SMRegistry& registry) 
    {
        // create the new vertex on the longest edge (halfway between the two)
        shared_ptr<SMVertex> opposite_vertex = longest_edge.first;
        shared_ptr<SMVertex> normal_vertex = longest_edge.second;
        this->split_vertex = registry.get_or_create_vertex(*normal_vertex, *opposite_vertex, 0.5);

        // In 2D, we can just make the children directly, all simplices are line segments (and have same normal)
        // Additionally, we can just point the normal from the split vertex to the normal vertex
        if (is_2d())
        {
            this->normal_child = make_shared<SMSimplex>(
                dim, vector<shared_ptr<SMVertex>>{this->split_vertex, normal_vertex}, depth+1);
            this->opposite_child = make_shared<SMSimplex>(
                dim, vector<shared_ptr<SMVertex>>{this->split_vertex, opposite_vertex}, depth+1);
            Vec opposite_to_normal = normal_vertex->weight - opposite_vertex->weight;
            this->splitting_hyperplane_normal = make_shared<Vec>(opposite_to_normal.normalised());
            return;
        }
         
        // Create vector of all vertices common to both children
        vector<shared_ptr<SMVertex>> common_vertices;
        common_vertices.push_back(this->split_vertex);
        for (shared_ptr<SMVertex> vertex : vertices) {
            if ((*vertex != *normal_vertex) && (*vertex != *opposite_vertex))
            {
                common_vertices.push_back(vertex);
            }
        }

        // Compute normal (using the dim-1 many common points of the child simplices)
        this->splitting_hyperplane_normal = this->compute_hyperplane_normal(child_common_simplex_vertices);

        // and make sure that the normal points towards the normal side child
        Vec splitting_edge_normal_dir = (normal_vertex->weight - opposite_vertex->weight);
        if (splitting_edge_normal_dir.dot(this->splitting_hyperplane_normal) < 0.0) {
            this->splitting_hyperplane_normal *= -1.0;
        }

        // Normal side child simplex
        shared_ptr<vector<shared_ptr<SMVertex>>> normal_side_child_vertices = make_shared<vector<shared_ptr<SMVertex>>>(
            common_vertices);
        normal_side_child_vertices->push_back(normal_vertex);
        this->normal_child = make_shared<SMSimplex>(dim, normal_side_child_vertices, depth+1);

        // Opposite side child simplex
        shared_ptr<vector<shared_ptr<SMVertex>>> opposite_side_child_vertices = make_shared<vector<shared_ptr<SMVertex>>>(
            common_vertices);
        opposite_side_child_vertices->push_back(opposite_vertex);
        this->opposite_child = make_shared<SMSimplex>(dim, opposite_side_child_vertices, depth+1);
    }
    /**
     * return true if point is on normal side of the plane defined by halfplane_point and halfplane_normal 
     (i.e. if wegith-halfplane_point dot halfplane_normal == 0)
    */
    bool SMSimplex::halfplane_check(
        const Vec& halfplane_point, 
        const Vec& halfplane_normal, 
        const Vec& weight) const 
    {
        Vec diff = weight - halfplane_point;
        return diff.dot(halfplane_normal) >= 0;
    }

    /**
     * Travers this node to child node
     */
    shared_ptr<SMSimplex> SMSimplex::traverse(const Vec& weight) const 
    {
        if (this->halfplane_check(this->split_vertex->weight, this->splitting_hyperplane_normal, weight)) {
            return this->normal_child;
        } else {
            return this->opposite_child;
        }
    }

    /**
     * If this node is a leaf in the binary tree
     */
    bool SMSimplex::is_leaf() const 
    {
        return this->normal_child == nullptr && this->opposite_child == nullptr;
    }

    /**
     * Checks if simplex is potentially worth subdividing
     */
    bool SMSimplex::vertexes_contain_multiple_unique_values() const
    {
        for (shared_ptr<SMVertex> vertex : vertices) {
            if (vertex->value_estimate != vertices[0]->value_estimate) {
                return true;
            }
        }
        return false;
    }

    bool SMSimplex::allowed_to_subdivide(double min_radius, int max_depth, int split_counter_threshold) const
    {
        if (depth >= max_depth) {
            return false;
        }
        if (radius <= min_radius) {
            return false;
        }
        return true;
    }

    bool SMSimplex::should_subdivide(double min_radius, int max_depth, int split_counter_threshold) const
    {
        // If already subdivided, no need
        if (!is_leaf()) {
            return false;
        }

        // If not allowed to subdivide, return false
        if (!allowed_to_subdivide(min_radius, max_depth)) {
            return false;
        }

        // If vertexes all share same value estimate, reset counter and no need to subdivide
        if (!vertexes_contain_multiple_unique_values()) {
            split_counter = 0;
            return false;
        }

        // increment counter
        split_counter++;

        // if counter is greater than threshold, then we should subdivide
        return split_counter >= split_counter_threshold;
    }
}


// ------------------------------------------------------------
// SMEdge
// ------------------------------------------------------------


namespace thts {

    SMEdge::SMEdge(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1) : 
        v0(v0), 
        v1(v1),
        midpoint(nullptr),
        child_edge_0(nullptr),
        child_edge_1(nullptr)
    {
    }

    size_t SMEdge::hash() const
    {
        return thts::helper::unordered_hash(*v0,*v1);
    }

    bool SMEdge::equals(const SMEdge& other) const
    {
        return ((v0->equals(*other.v0) && v1->equals(*other.v1))
            || (v0->equals(*other.v1) && v1->equals(*other.v0)));
    }

    bool SMEdge::operator==(const SMEdge& other) const
    {
        return equals(other);
    }

    bool SMEdge::operator!=(const SMEdge& other) const
    {
        return !equals(other);
    }

    void SMEdge::split(SMRegistry& registry)
    {
        this->midpoint = registry.get_or_create_vertex(v0, v1, 0.5);
        this->child_edge_0 = registry.get_or_create_edge(v0, this->midpoint);
        this->child_edge_1 = registry.get_or_create_edge(this->midpoint, v1);
    }

    shared_ptr<unordered_set<shared_ptr<SMEdge>>> SMEdge::get_edge_partition() const
    {
        shared_ptr<unordered_set<shared_ptr<SMEdge>>> partition = make_shared<unordered_set<shared_ptr<SMEdge>>>();
        get_edge_partition_helper(shared_from_this(), *partition);
        return partition;
    }

    void SMEdge::get_edge_partition_helper(std::shared_ptr<SMEdge> edge, std::unordered_set<std::shared_ptr<SMEdge>>& partition) const
    {
        if (edge->child_edge_0 != nullptr && edge->child_edge_1 != nullptr) 
        {
            get_edge_partition_helper(edge->child_edge_0, partition);
            get_edge_partition_helper(edge->child_edge_1, partition);
            return;
        }
        partition.insert(edge);
    }
}




// ------------------------------------------------------------
// SMRegistry
// ------------------------------------------------------------

namespace thts {

    shared_ptr<SMVertex> SMRegistry::get_or_create_vertex(const Vec& weight, const Vec& value_estimate, double entropy_estimate=0.0)
    {  
        // Create a new vertex and lookup the unique version of it
        shared_ptr<SMVertex> vertex = make_shared<SMVertex>(weight, value_estimate, entropy_estimate);
        return this->lookup_unique_vertex(vertex);
    }

    shared_ptr<SMVertex> SMRegistry::get_or_create_vertex(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1, double ratio=0.5)
    {
        // Create a new vertex and lookup the unique version of it
        shared_ptr<SMVertex> vertex = make_shared<SMVertex>(v0, v1, ratio);
        return this->lookup_unique_vertex(vertex);
    }

    shared_ptr<SMEdge> SMRegistry::get_or_create_edge(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1)
    {
        // Create a new edge and lookup the unique version of it
        shared_ptr<SMEdge> edge = make_shared<SMEdge>(v0, v1);
        return this->lookup_unique_edge(edge);
    }

    shared_ptr<SMVertex> SMRegistry::lookup_unique_vertex(shared_ptr<SMVertex> vertex)
    {
        if (vertex_map.contains(vertex)) {
            return vertex_map.at(vertex);
        }
        // new vertex, add to map and return it
        vertex_map[vertex] = vertex;
        return vertex;
    }

    shared_ptr<SMEdge> SMRegistry::lookup_unique_edge(shared_ptr<SMEdge> edge)
    {
        if (edge_map.contains(edge)) {
            return edge_map.at(edge);
        }
        // new edge, add to map and return it
        edge_map[edge] = edge;
        return edge;
    }
}




// ------------------------------------------------------------
// SMMesh
// ------------------------------------------------------------

namespace thts {

    // Constructor
    SMMesh::SMMesh(int dim) : 
        dim(dim), 
        registry(), 
        root_simplex(nullptr), 
        all_vertices_set(),
        all_vertices_vector(),
        simplex_to_edge_map(),
        edge_to_simplex_map(),
        non_conforming_simplices(), 
        non_conforming_simplices_by_depth()
    {
    }

    /**
     * Destructor needs to make sure that the SMVertex graph gets cleaned up 
     Worth noting that it is a graph of shared pointers
     So if we dont explicitly call reset on the neighbour maps, then the memory will not be freed
     */
    SMMesh::~SMMesh() 
    {
        for (shared_ptr<SMVertex> vertex : this->all_vertices_vector) {
            vertex->neighbours.reset();
        }
    }

    // Initialise the mesh
    void SMMesh::initialise_mesh(Vec& heuristic_value_estimate)
    {
        // Initialise the root simplex as the unit simplex
        vector<shared_ptr<SMVertex>> unit_simplex_vertices;
        for (int i=0; i<dim; i++) {
            Vec basis_vector = Vec(dim, 0.0);
            basis_vector.vec[i] = 1.0;
            shared_ptr<SMVertex> simplex_vertex = registry.get_or_create_vertex(basis_vector, heuristic_value_estimate, 0.0);
            unit_simplex_vertices.push_back(simplex_vertex);
        }
        this->root_simplex = make_shared<SMSimplex>(dim, unit_simplex_vertices, 0);

        // Initialise the all_vertices_set and all_vertices_vector
        for (shared_ptr<SMVertex> vertex : unit_simplex_vertices) {
            this->all_vertices_set.insert(vertex);
            this->all_vertices_vector.push_back(vertex);
        }

        // In 2D we don't need to worry about non-conforming simplices or the mesh graph, so we are done
        if (is_2d()) {
            return;
        }

        // Create an SMEdge for each pair of vertices
        // Add them to the mesh graph, mapping to and from the root simplex
        // And connect the SMVertices to each other
        for (size_t i=0; i<unit_simplex_vertices.size(); i++) {
            for (size_t j=i+1; j<unit_simplex_vertices.size(); j++) {
                shared_ptr<SMEdge> edge = registry.get_or_create_edge(
                    unit_simplex_vertices[i], unit_simplex_vertices[j]);
                this->edge_to_simplex_map[edge].insert(this->root_simplex);
                this->simplex_to_edge_map[this->root_simplex].insert(edge);
                unit_simplex_vertices[i]->add_bidirectional_connection(unit_simplex_vertices[j]);
            }
        }
    }

    shared_ptr<SMVertex> SMMesh::sample_random_vertex() const
    {
        int rand_index = rand_manager.get_rand_int(0, this->all_vertices_vector.size());
        return this->all_vertices_vector.at(rand_index);
    }

    shared_ptr<SMSimplex> SMMesh::get_simplex(const Vec& weight) const
    {
        shared_ptr<SMSimplex> cur = this->root_simplex;
        while (!cur->is_leaf()) {
            cur = cur->traverse(weight);
        }
        return cur;
    }

    shared_ptr<SMVertex> SMMesh::get_closest_vertex(const Vec& weight) const
    {
        shared_ptr<SMSimplex> simplex = get_simplex(weight);
        return this->get_closest_vertex(simplex, weight);
    }

    shared_ptr<SMVertex> SMMesh::get_closest_vertex(shared_ptr<SMSimplex> simplex, const Vec& weight) const
    {
        return simplex->get_closest_vertex(weight);
    }

    int SMMesh::get_num_updates(shared_ptr<SMVertex> vertex) const
    {
        return vertex->num_updates;
    }

    Vec SMMesh::get_value_estimate(shared_ptr<SMVertex> vertex) const
    {
        return vertex->value_estimate;
    }

    Vec SMMesh::get_value_estimate_for_search(shared_ptr<SMVertex> vertex) const
    {
        return vertex->value_estimate_for_search;
    }

    double SMMesh::get_entropy_estimate(shared_ptr<SMVertex> vertex) const
    {
        return vertex->entropy_estimate;
    }

    void SMMesh::update_vertex_values_and_share(
        shared_ptr<SMVertex> vertex, 
        int max_push_radius, 
        const Vec& value_estimate, 
        const Vec& value_estimate_for_search, 
        double entropy_estimate)
    {
        vertex->num_updates++;
        vertex->value_estimate = value_estimate;
        vertex->value_estimate_for_search = value_estimate_for_search;
        vertex->entropy_estimate = entropy_estimate;

        vertex->share_values_message_passing(max_push_radius);
    }

    void SMMesh::maybe_subdivide(shared_ptr<SMSimplex> simplex)
    {
        // Perform the subdivision if needed
        if (simplex->should_subdivide(min_radius, max_depth, split_counter_threshold)) {
            this->subdivide_simplex(simplex);
        }

        // We are done if there are no non-conforming simplices
        if (this->non_conforming_simplices.empty()) {
            return;
        }

        // Get the lowest depth non-conforming simplex
        shared_ptr<SMSimplex> lowest_depth_non_conforming_simplex = this->pop_lowest_depth_non_conforming_simplex();
        this->subdivide_simplex(lowest_depth_non_conforming_simplex);
    }

    std::string SMMesh::get_pretty_print_string() const
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

    ConvexHull SMMesh::get_approximate_convex_hull() const
    {
        unordered_set<Vec> ch_points;
        for (shared_ptr<SMVertex> vertex : this->all_vertices_vector) {
            ch_points.insert(vertex->value_estimate);
        }
        return ConvexHull(ch_points);
    }

    bool SMMesh::is_2d() const
    {
        return this->dim == 2;
    }

    // Helper to get the lowest depth non-conforming simplex
    // Non conforming simplices are stored in a map by depth, so we can just get the first one
    shared_ptr<SMSimplex> SMMesh::pop_lowest_depth_non_conforming_simplex()
    {
        shared_ptr<SMSimplex> non_conforming_simplex = (*this->non_conforming_simplices_by_depth.begin()).second.pop();
        this->non_conforming_simplices.erase(non_conforming_simplex);
    }

    void SMMesh::subdivide_simplex(shared_ptr<SMSimplex> simplex)
    {
        // First get the simplex to create its children
        simplex->create_children(this->registry);

        // If 2d, then we are actually done
        if (is_2d()) {
            return;
        }

        // Get the edge corresponding to the longest edge of the simplex
        shared_ptr<SMVertex> longest_edge_vertex_0 = simplex->longest_edge.first;
        shared_ptr<SMVertex> longest_edge_vertex_1 = simplex->longest_edge.second;
        shared_ptr<SMEdge> longest_edge = this->registry.get_or_create_edge(
            longest_edge_vertex_0, longest_edge_vertex_1);

        // Check if this edge is currently in the mesh graph
        // If it is not, then it has already been split by a previous subdivision
        if (this->edge_to_simplex_map.contains(longest_edge)) 
        {
            // Split the edge to create two child edges
            longest_edge->split(this->registry);
            shared_ptr<SMEdge> child_edge_0 = longest_edge->child_edge_0;
            shared_ptr<SMEdge> child_edge_1 = longest_edge->child_edge_1;

            // insert these new edges into the mesh graph, by inheriting connections from the parent edge
            this->inherit_parent_edge_connections(child_edge_0, longest_edge);
            this->inherit_parent_edge_connections(child_edge_1, longest_edge);

            // Remove the parent edge from the mesh graph
            this->remove_edge_from_mesh(longest_edge);

            // Update the graph of vertices for the new edges and removal of the parent edge
            shared_ptr<SMVertex> longest_edge_v0 = longest_edge->v0;
            shared_ptr<SMVertex> longest_edge_v1 = longest_edge->v1;
            shared_ptr<SMVertex> longest_edge_midpoint = longest_edge->midpoint;

            longest_edge_v0->erase_bidirectional_connection(longest_edge_v1);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v0);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v1);

            // Update non-conformity for the new edges
            this->update_non_conformity_for_new_edge(child_edge_0);
            this->update_non_conformity_for_new_edge(child_edge_1);
        }

        // Now remove the parent simplex from the mesh graph
        // And add the children simplices to the mesh graph
        this->remove_simplex_from_mesh_graph(simplex);
        this->add_new_simplex_to_mesh_graph(simplex->normal_child);
        this->add_new_simplex_to_mesh_graph(simplex->opposite_child);
    }

    void SMMesh::inherit_parent_edge_connections(shared_ptr<SMEdge> new_edge, shared_ptr<SMEdge> parent_edge)
    {
        unordered_set<shared_ptr<SMSimplex>> adjacent_simplices = this->edge_to_simplex_map.at(parent_edge);
        this->edge_to_simplex_map[new_edge] = adjacent_simplices;
    }

    void SMMesh::update_non_conformity_for_new_edge(shared_ptr<SMEdge> new_edge)
    {
        unordered_set<shared_ptr<SMSimplex>> adjacent_simplices = this->edge_to_simplex_map.at(new_edge);
        for (shared_ptr<SMSimplex> simplex : adjacent_simplices) {
            if (!simplex->contains_vertex(new_edge->v0) || !simplex->contains_vertex(new_edge->v1)) {
                simplex->is_non_conforming = true;
                this->non_conforming_simplices.insert(simplex);
                this->non_conforming_simplices_by_depth[simplex->depth].push(simplex);
            }
        }
    }

    void SMMesh::remove_edge_from_mesh_graph(shared_ptr<SMEdge> edge)
    {
        // Remove pointers to this edge
        for (shared_ptr<SMSimplex> simplex : this->edge_to_simplex_map.at(edge)) 
        {  
            // Removes the edge from the simplex's set of edges
            this->simplex_to_edge_map.at(simplex).erase(edge);
        }
        // And then remove this edge (remove the entire entry from edge, so the entire set of simplices is removed)
        this->edge_to_simplex_map.erase(edge);
    }

    void SMMesh::remove_simplex_from_mesh_graph(shared_ptr<SMSimplex> simplex)
    {
        // Remove pointers to this simplex
        for (shared_ptr<SMEdge> edge : this->simplex_to_edge_map.at(simplex)) 
        {
            // Removes the simplex from the edge's set of simplices
            this->edge_to_simplex_map.at(edge).erase(simplex);
        }
        // And then remove this simplex (remove the entire entry from simplex, so the entire set of edges is removed)
        this->simplex_to_edge_map.erase(simplex);
    }

    void SMMesh::add_new_simplex_to_mesh_graph(shared_ptr<SMSimplex> simplex)
    {
        // Get edges of simplex
        unordered_set<shared_ptr<SMEdge>> edges;
        for (size_t i=0; i<simplex->vertices.size(); i++) {
            for (size_t j=i+1; j<simplex->vertices.size(); j++) {
                shared_ptr<SMEdge> edge = this->registry.get_or_create_edge(simplex->vertices[i], simplex->vertices[j]);
                edges.insert(edge);
            }
        }

        // Get edges in mesh graph along the same lines as simplex
        unordered_set<shared_ptr<SMEdge>> mesh_graph_edges;
        for (shared_ptr<SMEdge> edge : edges) {
            unordered_set<shared_ptr<SMEdge>> edge_partition = edge->get_edge_partition();
            for (shared_ptr<SMEdge> edge_partition_edge : edge_partition) {
                mesh_graph_edges.insert(edge_partition_edge);
            }
        }

        // Update non-conformity for the new simplex
        // This new simplex is only conforming if all of its edges are in the mesh graph
        // Because mesh graph should not contain any overlapping edges, we can just compare the sizes
        if (edges.size() != mesh_graph_edges.size()) {
            simplex->is_non_conforming = true;
            this->non_conforming_simplices.insert(simplex);
            this->non_conforming_simplices_by_depth[simplex->depth].push(simplex);
        }

        // For each mesh edge, add pointers to this simplex
        for (shared_ptr<SMEdge> edge : mesh_graph_edges) {
            this->edge_to_simplex_map[edge].insert(simplex);
        }

        // And add the pointers to mesh edges from this simplex
        this->simplex_to_edge_map[simplex] = mesh_graph_edges;
    }
}






























































    void TN::_ensure_neighbourhood_graph_connected() 
    {
        for (size_t i=0; i<simplex_vertices->size(); i++) {
            for (size_t j=i+1; j<simplex_vertices->size(); j++) {
                simplex_vertices->at(i)->add_connection(simplex_vertices->at(j));
            }
        }
    }
    
    
    
    



    shared_ptr<TN> SimplexMap::get_leaf_tn_node(const Eigen::ArrayXd& ctx) const 
    {
        shared_ptr<TN> cur = root_node;
        while (cur->has_children()) {
            cur = cur->get_child(ctx);
        }
        return cur;
    }

    shared_ptr<NGV> SimplexMap::sample_random_ngv_vertex(RandManager& rand_manager) const
    {
        int rand_index = rand_manager.get_rand_int(0,n_graph_vertices->size());
        return n_graph_vertices->at(rand_index);
    }





























    





// ------------------------------------------------------------
// Allowing structs to be used in unordered_set and unordered_map
// ------------------------------------------------------------
namespace std {
    using namespace thts;

    // SMVertex
    size_t hash<SMVertex>::operator()(const SMVertex& v) const 
    {
        return v.hash();
    }

    size_t equal_to<SMVertex>::operator()(const SMVertex& v0, const SMVertex& v1) const 
    {
        return v0.equals(v1);
    }

    template<>
    bool operator==(const SMVertex& v0, const SMVertex& v1) 
    {
        return v0.equals(v1);
    }

    // shared_ptr<SMVertex>
    size_t hash<shared_ptr<SMVertex>>::operator()(const shared_ptr<SMVertex>& v) const 
    {
        return v->hash();
    }

    size_t equal_to<shared_ptr<SMVertex>>::operator()(const shared_ptr<SMVertex>& v0, const shared_ptr<SMVertex>& v1) const 
    {
        return v0->equals(*v1);
    }

    template<>
    bool operator==(const shared_ptr<SMVertex>& v0, const shared_ptr<SMVertex>& v1) 
    {
        return v0->equals(*v1);
    }

    // SMEdge
    size_t hash<SMEdge>::operator()(const SMEdge& e) const
    {
        return e.hash();
    }

    size_t equal_to<SMEdge>::operator()(const SMEdge& e0, const SMEdge& e1) const
    {
        return e0.equals(e1);
    }

    template<>
    bool operator==(const SMEdge& e0, const SMEdge& e1) 
    {
        return e0.equals(e1);
    }

    // shared_ptr<SMEdge>
    size_t hash<shared_ptr<SMEdge>>::operator()(const shared_ptr<SMEdge>& e) const 
    {
        return e->hash();
    }
    
    size_t equal_to<shared_ptr<SMEdge>>::operator()(const shared_ptr<SMEdge>& e0, const shared_ptr<SMEdge>& e1) const 
    {
        return e0->equals(*e1);
    }

    template<>
    bool operator==(const shared_ptr<SMEdge>& e0, const shared_ptr<SMEdge>& e1) 
    {
        return e0->equals(*e1);
    }
}