#include "mo/data_structures/simplex_map.h"

#include "helper_templates.h"
#include "mo/mo_helper.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iterator>
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
        cached_hash(std::hash<Vec>()(this->weight)),
        num_direct_updates(0),
        num_updates(0),
        value_estimate(Vec::Zero(weight.dim())),
        value_estimate_local(heuristic_value_estimate),
        entropy_estimate(entropy_estimate),
        neighbours(make_shared<unordered_set<shared_ptr<SMVertex>>>())
    {
    }

    /**
    * Constructor as midpoint of two other vertices
    */
    SMVertex::SMVertex(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1, double ratio) : 
        weight(ratio * v0->weight + (1.0-ratio) * v1->weight),
        cached_hash(std::hash<Vec>()(this->weight)),
        num_direct_updates(0),
        num_updates(1), // count this initialisation as a message passing update
        value_estimate(v0->value_estimate),
        value_estimate_local(v0->value_estimate_local),
        entropy_estimate(v0->entropy_estimate),
        neighbours(make_shared<unordered_set<shared_ptr<SMVertex>>>())
    {
        if (v1->value_estimate.dot(this->weight) > v0->value_estimate.dot(this->weight)) {
            this->value_estimate = v1->value_estimate;
            this->value_estimate_local = v1->value_estimate_local;
            this->entropy_estimate = v1->entropy_estimate;
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
        return cached_hash;
    };

    bool SMVertex::equals(const SMVertex& other) const 
    {
        return weight.equals(other.weight);
    };

    bool SMVertex::operator==(const SMVertex& other) const 
    {
        return equals(other);
    };

    bool SMVertex::operator!=(const SMVertex& other) const 
    {
        return !equals(other);
    };
    
    /**
    * Message passing (BFS)
    */
    void SMVertex::share_values_message_passing_subset(RandManager& rand_manager, int max_neighbours_to_push_to)
    {
        const unordered_set<shared_ptr<SMVertex>>* vertices_to_push_to = neighbours.get();
        unique_ptr<unordered_set<shared_ptr<SMVertex>>> subsample_vertices;
        if (max_neighbours_to_push_to != -1 && static_cast<size_t>(max_neighbours_to_push_to) < neighbours->size()) {
            subsample_vertices = make_unique<unordered_set<shared_ptr<SMVertex>>>();
            std::sample(
                neighbours->begin(),
                neighbours->end(),
                std::inserter(*subsample_vertices, subsample_vertices->begin()),
                static_cast<size_t>(max_neighbours_to_push_to),
                rand_manager.get_random_device());
            vertices_to_push_to = subsample_vertices.get();
        }
        for (const shared_ptr<SMVertex>& vertex : *vertices_to_push_to) {
            share_values_message_passing_helper(*this, *vertex);
        }
    }

    void SMVertex::share_values_message_passing(RandManager& rand_manager, int max_push_radius, int max_neighbours_to_push_to) 
    {  
        if (max_push_radius != 1 && max_neighbours_to_push_to != -1) {
            throw std::invalid_argument("max_push_radius != 1 is not supported unless max_neighbours_to_push_to is set "
                "to -1 (push to all neighbours). I.e. you can only push to a subset of neighbours, or, push in a wider "
                "radius, but not both");
        }

        if (max_neighbours_to_push_to != -1)
        {
            return this->share_values_message_passing_subset(rand_manager, max_neighbours_to_push_to);
        }

        int current_push_radius = 0;
        deque<shared_ptr<SMVertex>> vertex_queue;
        deque<shared_ptr<SMVertex>> next_vertex_queue;
        unordered_set<shared_ptr<SMVertex>> visited_vertices;
        vertex_queue.push_back(shared_from_this());
        visited_vertices.insert(shared_from_this());

        while (!vertex_queue.empty() && current_push_radius < max_push_radius) {
            shared_ptr<SMVertex> current_vertex = vertex_queue.front();
            vertex_queue.pop_front();
            for (const shared_ptr<SMVertex>& neighbour_ptr : *current_vertex->neighbours) {
                // insert returns {iterator, inserted}; if not inserted, we've already visited
                auto [vit, inserted] = visited_vertices.insert(neighbour_ptr);
                if (!inserted) continue;
                bool success = share_values_message_passing_helper(*current_vertex, *neighbour_ptr);
                if (success) 
                {
                    next_vertex_queue.push_back(neighbour_ptr);
                }
            }
            if (vertex_queue.empty()) {
                current_push_radius++;
                vertex_queue = next_vertex_queue;
                next_vertex_queue = deque<shared_ptr<SMVertex>>();
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
        // Dont propogate fictitious values from heuristics
        if (from_vertex.num_updates <= 0)
        {
            return false;
        }

        // we are attempting to update to_vertex, so increment the number of updates
        to_vertex.num_updates += 1;

        // But only update values if it is actually an improvement
        if (from_vertex.value_estimate.dot(to_vertex.weight) > to_vertex.value_estimate.dot(to_vertex.weight)) 
        {
            to_vertex.value_estimate = from_vertex.value_estimate;
            to_vertex.value_estimate_local = from_vertex.value_estimate_local;
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
        vertices_set(vertices.begin(), vertices.end()),
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
            Vec& v0 = this->vertices[0]->weight;
            Vec& v1 = this->vertices[1]->weight;
            if (v0[0] > v1[0]) 
            {
                this->longest_edge = std::make_pair(this->vertices[1], this->vertices[0]);
            }
            else
            {
                this->longest_edge = std::make_pair(this->vertices[0], this->vertices[1]);
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

    bool SMSimplex::is_2d() const
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
        for (const shared_ptr<SMVertex>& vertex : vertices) {
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
     Returns the SMVertex used to split this simplex
     (May already exist in registry from a previous split)
    */
    shared_ptr<SMVertex> SMSimplex::create_children(SMRegistry& registry) 
    {
        // create the new vertex on the longest edge (halfway between the two)
        shared_ptr<SMVertex> opposite_vertex = longest_edge.first;
        shared_ptr<SMVertex> normal_vertex = longest_edge.second;
        this->split_vertex = registry.get_or_create_vertex(normal_vertex, opposite_vertex, 0.5);

        // In 2D, we can just make the children directly, all simplices are line segments (and have same normal)
        // Additionally, we can just point the normal from the split vertex to the normal vertex
        if (is_2d())
        {
            vector<shared_ptr<SMVertex>> normal_side_vertices = {this->split_vertex, normal_vertex};
            vector<shared_ptr<SMVertex>> opposite_side_vertices = {this->split_vertex, opposite_vertex};
            this->normal_child = make_shared<SMSimplex>(dim, normal_side_vertices, depth+1);
            this->opposite_child = make_shared<SMSimplex>(dim, opposite_side_vertices, depth+1);
            Vec opposite_to_normal = normal_vertex->weight - opposite_vertex->weight;
            this->splitting_hyperplane_normal = make_shared<Vec>(opposite_to_normal.normalised());
            return this->split_vertex;
        }
         
        // Create vector of all vertices common to both children.
        // N.B. we compare raw pointer identity here rather than the
        // overloaded shared_ptr operator!= (which dispatches to
        // SMVertex::equals and an O(dim) Vec comparison). SMRegistry
        // canonicalises SMVertex instances, so pointer identity is
        // exactly the right semantic and is O(1).
        vector<shared_ptr<SMVertex>> common_vertices;
        common_vertices.push_back(this->split_vertex);
        SMVertex* normal_raw = normal_vertex.get();
        SMVertex* opposite_raw = opposite_vertex.get();
        for (const shared_ptr<SMVertex>& vertex : vertices) {
            SMVertex* vraw = vertex.get();
            if (vraw != normal_raw && vraw != opposite_raw)
            {
                common_vertices.push_back(vertex);
            }
        }

        // Compute normal (using the dim-1 many common points of the child simplices)
        this->splitting_hyperplane_normal = make_shared<Vec>(this->compute_hyperplane_normal(common_vertices));

        // and make sure that the normal points towards the normal side child
        Vec splitting_edge_normal_dir = (normal_vertex->weight - opposite_vertex->weight);
        Vec& hyperplane_normal = *this->splitting_hyperplane_normal;
        if (splitting_edge_normal_dir.dot(hyperplane_normal) < 0.0) {
            this->splitting_hyperplane_normal = make_shared<Vec>(hyperplane_normal * -1.0);
        }

        // Normal side child simplex
        vector<shared_ptr<SMVertex>> normal_side_child_vertices(common_vertices);
        normal_side_child_vertices.push_back(normal_vertex);
        this->normal_child = make_shared<SMSimplex>(dim, normal_side_child_vertices, depth+1);

        // Opposite side child simplex
        vector<shared_ptr<SMVertex>> opposite_side_child_vertices(common_vertices);
        opposite_side_child_vertices.push_back(opposite_vertex);
        this->opposite_child = make_shared<SMSimplex>(dim, opposite_side_child_vertices, depth+1);

        return this->split_vertex;
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
        if (this->halfplane_check(this->split_vertex->weight, *this->splitting_hyperplane_normal, weight)) 
        {
            return this->normal_child;
        } 
        // else {
        return this->opposite_child;
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
        const Vec& first_value = vertices[0]->value_estimate;
        for (const shared_ptr<SMVertex>& vertex : vertices) {
            if (vertex->value_estimate != first_value) {
                return true;
            }
        }
        return false;
    }

    bool SMSimplex::allowed_to_subdivide(
        double min_radius, 
        int max_depth) const
    {
        if (depth >= max_depth) {
            return false;
        }
        if (radius <= min_radius) {
            return false;
        }
        return true;
    }

    bool SMSimplex::should_subdivide(
        double min_radius, 
        int max_depth, 
        int split_counter_threshold, 
        bool always_allow_non_conforming_simplex_to_split) const
    {
        // If already subdivided, no need
        if (!is_leaf()) {
            return false;
        }

        // If always splitting non-conforming simplices, then we can subdivide
        if (this->is_non_conforming && always_allow_non_conforming_simplex_to_split) {
            return true;
        }

        // If not allowed to subdivide, return false
        if (!this->allowed_to_subdivide(min_radius, max_depth)) {
            return false;
        }

        // If we are allowed to subdivide, and non conforming, we can subdivide and skip the checks
        if (this->is_non_conforming) {
            return true;
        }

        // If vertexes all share same value estimate, reset counter and no need to subdivide
        if (!this->vertexes_contain_multiple_unique_values()) {
            split_counter = 0;
            return false;
        }

        // increment counter
        this->split_counter++;

        // if counter is greater than threshold, then we should subdivide
        return this->split_counter >= split_counter_threshold;
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
        child_edge_1(nullptr),
        cached_hash(thts::helper::unordered_hash(*v0, *v1))
    {
    }

    size_t SMEdge::hash() const
    {
        return cached_hash;
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

    bool SMEdge::split(SMRegistry& registry)
    {
        // If already split, return false
        if (this->midpoint != nullptr) {
            return false;
        }

        // Create the midpoint vertex and half edges
        this->midpoint = registry.get_or_create_vertex(this->v0, this->v1, 0.5);
        this->child_edge_0 = registry.get_or_create_edge(this->v0, this->midpoint);
        this->child_edge_1 = registry.get_or_create_edge(this->midpoint, this->v1);
        return true;
    }

    unordered_set<shared_ptr<SMEdge>> SMEdge::get_edge_partition() const
    {
        // Return by value; RVO/NRVO avoids a copy, and we skip a heap
        // allocation (previously the set was wrapped in a shared_ptr).
        unordered_set<shared_ptr<SMEdge>> partition;
        this->get_edge_partition_helper(std::const_pointer_cast<SMEdge>(shared_from_this()), partition);
        return partition;
    }

    void SMEdge::get_edge_partition_helper(const std::shared_ptr<SMEdge>& edge, std::unordered_set<std::shared_ptr<SMEdge>>& partition) const
    {
        if (edge->child_edge_0 != nullptr && edge->child_edge_1 != nullptr) 
        {
            get_edge_partition_helper(edge->child_edge_0, partition);
            get_edge_partition_helper(edge->child_edge_1, partition);
            return;
        }
        partition.insert(edge);
    }

    Vec SMEdge::find_closest_point_on_edge(const Vec& point) const
    {
        double ratio = this->find_closest_point_on_edge_ratio(point);
        return this->v0->weight * (1.0-ratio) + this->v1->weight * ratio;
    }

    // Projects v0->point onto v0->v1 and returns the ratio of the projection
    // This gives the value of t for the closest point on the edge
    // u = v0 + t * (v1 - v0)
    double SMEdge::find_closest_point_on_edge_ratio(const Vec& point) const
    {
        Vec v0_to_point = point - this->v0->weight;
        Vec v0_to_v1 = this->v1->weight - this->v0->weight;
        double ratio = v0_to_point.dot(v0_to_v1) / v0_to_v1.dot(v0_to_v1);
        if (ratio < 0.0)
        {
            return 0.0;
        }
        if (ratio > 1.0)
        {
            return 1.0;
        }
        return ratio;
    }
}




// ------------------------------------------------------------
// SMRegistry
// ------------------------------------------------------------

namespace thts {

    shared_ptr<SMVertex> SMRegistry::get_or_create_vertex(const Vec& weight, const Vec& value_estimate, double entropy_estimate)
    {
        // Create a new vertex and lookup the unique version of it
        shared_ptr<SMVertex> vertex = make_shared<SMVertex>(weight, value_estimate, entropy_estimate);
        return this->lookup_unique_vertex(vertex);
    }

    shared_ptr<SMVertex> SMRegistry::get_or_create_vertex(shared_ptr<SMVertex> v0, shared_ptr<SMVertex> v1, double ratio)
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
        // Single hash lookup: find() returns an iterator that we can branch
        // on without paying for a second hash+probe via contains()/at().
        auto it = vertex_map.find(vertex);
        if (it != vertex_map.end()) {
            return it->second;
        }
        vertex_map.emplace(vertex, vertex);
        return vertex;
    }

    shared_ptr<SMEdge> SMRegistry::lookup_unique_edge(shared_ptr<SMEdge> edge)
    {
        auto it = edge_map.find(edge);
        if (it != edge_map.end()) {
            return it->second;
        }
        edge_map.emplace(edge, edge);
        return edge;
    }
}




// ------------------------------------------------------------
// SMMesh
// ------------------------------------------------------------

namespace thts {

    // Constructor
    SMMesh::SMMesh(int dim, bool find_exact_closest_vertex, bool eventually_conforming_mesh, bool always_allow_non_conforming_simplex_to_split) : 
        dim(dim), 
        find_exact_closest_vertex(find_exact_closest_vertex),
        eventually_conforming_mesh(eventually_conforming_mesh),
        always_allow_non_conforming_simplex_to_split(always_allow_non_conforming_simplex_to_split),
        registry(), 
        root_simplex(nullptr), 
        all_vertices_set(),
        all_vertices_vector(),
        simplex_to_edge_map(),
        edge_to_simplex_map(),
        non_conforming_simplices(), 
        non_conforming_simplices_by_depth()
    {
        if (find_exact_closest_vertex && !eventually_conforming_mesh) {
            throw std::invalid_argument("find_exact_closest_vertex=true implementation assumes that eventually_conforming_mesh is true");
        }
    }

    /**
     * Destructor needs to make sure that the SMVertex graph gets cleaned up 
     Worth noting that it is a graph of shared pointers
     So if we dont explicitly call reset on the neighbour maps, then the memory will not be freed
     */
    SMMesh::~SMMesh() 
    {
        for (const shared_ptr<SMVertex>& vertex : this->all_vertices_vector) {
            vertex->neighbours.reset();
        }
    }

    // Initialise the mesh
    void SMMesh::initialise_mesh(Vec& heuristic_value_estimate)
    {
        // Initialise the root simplex as the unit simplex
        vector<shared_ptr<SMVertex>> unit_simplex_vertices;
        for (int i=0; i<dim; i++) {
            Vec basis_vector = Vec::Zero(dim);
            basis_vector.vec[i] = 1.0;
            shared_ptr<SMVertex> simplex_vertex = registry.get_or_create_vertex(basis_vector, heuristic_value_estimate, 0.0);
            unit_simplex_vertices.push_back(simplex_vertex);
        }
        this->root_simplex = make_shared<SMSimplex>(dim, unit_simplex_vertices, 0);

        // Initialise the all_vertices_set and all_vertices_vector
        for (const shared_ptr<SMVertex>& vertex : unit_simplex_vertices) {
            this->all_vertices_set.insert(vertex);
            this->all_vertices_vector.push_back(vertex);
        }

        // Create an SMEdge for each pair of vertices
        // And connect the SMVertices to each other
        for (size_t i=0; i<unit_simplex_vertices.size(); i++) {
            for (size_t j=i+1; j<unit_simplex_vertices.size(); j++) {
                shared_ptr<SMEdge> edge = registry.get_or_create_edge(
                    unit_simplex_vertices[i], unit_simplex_vertices[j]);
                unit_simplex_vertices[i]->add_bidirectional_connection(unit_simplex_vertices[j]);
            }
        }

        // In 2D we don't need to worry about non-conforming simplices or the mesh graph, so we are done
        // And optionally we can turn this logic off, by setting eventually_conforming_mesh to false
        if (is_2d() || !eventually_conforming_mesh) {
            return;
        }

        // Initialise the mesh graph maps if we are enforcing conformity
        for (size_t i=0; i<unit_simplex_vertices.size(); i++) {
            for (size_t j=i+1; j<unit_simplex_vertices.size(); j++) {
                shared_ptr<SMEdge> edge = registry.get_or_create_edge(
                    unit_simplex_vertices[i], unit_simplex_vertices[j]);
                this->edge_to_simplex_map[edge].insert(this->root_simplex);
                this->simplex_to_edge_map[this->root_simplex].insert(edge);
            }
        }
    }

    shared_ptr<SMVertex> SMMesh::sample_random_vertex(RandManager& rand_manager) const
    {
        int rand_index = rand_manager.get_rand_int(0, static_cast<int>(this->all_vertices_vector.size()));
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
        return this->get_closest_vertex_and_adjoining_simplex(weight).first;
    }

    SMVertexSMSimplexPair SMMesh::get_closest_vertex_and_adjoining_simplex(const Vec& weight) const
    {
        shared_ptr<SMSimplex> simplex = this->get_simplex(weight);
        return this->get_closest_vertex_and_adjoining_simplex(weight, simplex);
    }

    shared_ptr<SMVertex> SMMesh::get_closest_vertex(const Vec& weight, shared_ptr<SMSimplex> simplex) const
    {
        return this->get_closest_vertex_and_adjoining_simplex(weight, simplex).first;
    }

    SMVertexSMSimplexPair SMMesh::get_closest_vertex_and_adjoining_simplex(
        const Vec& weight, shared_ptr<SMSimplex> simplex) const
    {
        shared_ptr<SMVertex> closest_vertex = simplex->get_closest_vertex(weight);
        if (this->is_2d() || !this->find_exact_closest_vertex) 
        {
            return make_pair(closest_vertex, simplex);
        }
        
        // Get a list of additional vertices to check that are in simplices adjacent to the given simplex
        vector<SMVertexSMSimplexPair> additional_vertices;
        unordered_set<shared_ptr<SMSimplex>> adjacent_simplices;
        adjacent_simplices.insert(simplex);

        // Loop through edges of simplex
        for (const shared_ptr<SMEdge>& edge : this->simplex_to_edge_map.at(simplex)) 
        {
            // Then simplices adjacent to this edge
            for (const shared_ptr<SMSimplex>& adjacent_simplex : this->edge_to_simplex_map.at(edge)) 
            {
                // Single hash lookup: insert returns {it, inserted}. If
                // inserted is false we've already processed this simplex.
                auto [ait, inserted] = adjacent_simplices.insert(adjacent_simplex);
                if (!inserted)
                {
                    continue;
                }
                // Then add all vertices of this simplex to the list (that we haven't already checked)
                for (const shared_ptr<SMVertex>& vertex : adjacent_simplex->vertices) 
                {
                    if (simplex->contains_vertex(vertex))
                    {
                        continue;
                    }
                    additional_vertices.push_back(make_pair(vertex, adjacent_simplex));
                }
            }   
        }

        // Find the closest vertex from these points (we can initialise with the one we already found)
        double closest_dist = closest_vertex->weight.dist(weight);
        shared_ptr<SMSimplex> closest_simplex = simplex;
        for (SMVertexSMSimplexPair& vertex_simplex_pair : additional_vertices) 
        {
            shared_ptr<SMVertex> vertex = vertex_simplex_pair.first;
            shared_ptr<SMSimplex> simplex = vertex_simplex_pair.second;
            double dist = vertex->weight.dist(weight);
            if (dist < closest_dist) 
            {
                closest_dist = dist;
                closest_vertex = vertex;
                closest_simplex = simplex;
            }
        }
        return make_pair(closest_vertex, closest_simplex);
    }

    int SMMesh::get_num_updates(shared_ptr<SMVertex> vertex) const
    {
        return vertex->num_updates;
    }

    Vec SMMesh::get_value_estimate(shared_ptr<SMVertex> vertex) const
    {
        return vertex->value_estimate;
    }

    Vec SMMesh::get_value_estimate_local(shared_ptr<SMVertex> vertex) const
    {
        return vertex->value_estimate_local;
    }

    double SMMesh::get_entropy_estimate(shared_ptr<SMVertex> vertex) const
    {
        return vertex->entropy_estimate;
    }

    void SMMesh::update_vertex_values_and_share(
        RandManager& rand_manager,
        shared_ptr<SMVertex> vertex, 
        int max_push_radius,
        int max_neighbours_to_push_to, 
        const Vec& value_estimate, 
        const Vec& value_estimate_local, 
        double entropy_estimate)
    {
        vertex->num_direct_updates++;
        vertex->num_updates++;
        vertex->value_estimate = value_estimate;
        vertex->value_estimate_local = value_estimate_local;
        vertex->entropy_estimate = entropy_estimate;

        vertex->share_values_message_passing(rand_manager, max_push_radius, max_neighbours_to_push_to);
    }

    void SMMesh::maybe_subdivide(
        shared_ptr<SMSimplex> simplex, 
        double min_radius, 
        int max_depth, 
        int split_counter_threshold)
    {
        // Perform the subdivision if needed
        if (simplex->should_subdivide(min_radius, max_depth, split_counter_threshold, this->always_allow_non_conforming_simplex_to_split)) 
        {
            this->subdivide_simplex(simplex, min_radius, max_depth, split_counter_threshold);
        }

        // We are done if there are no non-conforming simplices
        // Or if we are not enforcing conformity
        if (this->non_conforming_simplices.empty() || !eventually_conforming_mesh) {
            return;
        }

        // Get the lowest depth non-conforming simplex
        shared_ptr<SMSimplex> lowest_depth_non_conforming_simplex = this->pop_lowest_depth_non_conforming_simplex();
        this->subdivide_simplex(lowest_depth_non_conforming_simplex, min_radius, max_depth, split_counter_threshold);
    }

    std::string SMMesh::get_pretty_print_string() const
    {
        stringstream ss;
        ss << "Simplex map pretty print: {" << endl;
        ss << "Weight // Value" << endl;
        for (const shared_ptr<SMVertex>& v : this->all_vertices_vector) {
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

    /**
     * Get convex hull from value estimates in this map
     But don't include vertices that haven't been updated from their heuristic value estimate
     */
    ConvexHull SMMesh::get_approximate_convex_hull() const
    {
        unordered_set<Vec> ch_points;
        for (const shared_ptr<SMVertex>& vertex : this->all_vertices_vector) {
            if (vertex->num_updates <= 0) 
            {
                continue;
            }
            ch_points.insert(vertex->value_estimate);
        }
        return ConvexHull(ch_points);
    }

    bool SMMesh::is_2d() const
    {
        return this->dim == 2;
    }

    // 

    // Helper to get the lowest depth non-conforming simplex
    // Non conforming simplices are stored in a map by depth, so we can just get the first one
    // Get the first non-conforming simplex at the lowest depth
    shared_ptr<SMSimplex> SMMesh::pop_lowest_depth_non_conforming_simplex()
    {
        // If empty, return nullptr
        if (this->non_conforming_simplices_by_depth.empty()) 
        {
            return nullptr;
        }

        // Get set the first depth
        auto it = this->non_conforming_simplices_by_depth.begin();
        std::unordered_set<std::shared_ptr<SMSimplex>>& non_conforming_simplex_queue = it->second;

        // "Pop" from the set
        shared_ptr<SMSimplex> non_conforming_simplex = *non_conforming_simplex_queue.begin();
        non_conforming_simplex_queue.erase(non_conforming_simplex);

        // If the set for this depth is now empty, then remove the depth from the map
        if (non_conforming_simplex_queue.empty()) {
            this->non_conforming_simplices_by_depth.erase(it);
        }

        // Remove simplex from complete set and return
        this->non_conforming_simplices.erase(non_conforming_simplex);
        return non_conforming_simplex;
    }

    void SMMesh::subdivide_simplex(
        shared_ptr<SMSimplex> simplex, 
        double min_radius, 
        int max_depth, 
        int split_counter_threshold)
    {
        // First get the simplex to create its children
        shared_ptr<SMVertex> split_vertex = simplex->create_children(this->registry);

        // Add the split vertex to sets (single hash lookup: insert returns
        // {it, inserted}).
        {
            auto [it, inserted] = this->all_vertices_set.insert(split_vertex);
            if (inserted)
            {
                this->all_vertices_vector.push_back(split_vertex);
            }
        }

        // Get the edge corresponding to the longest edge of the simplex
        shared_ptr<SMVertex> longest_edge_vertex_0 = simplex->longest_edge.first;
        shared_ptr<SMVertex> longest_edge_vertex_1 = simplex->longest_edge.second;
        shared_ptr<SMEdge> longest_edge = this->registry.get_or_create_edge(
            longest_edge_vertex_0, longest_edge_vertex_1);

        // Split the edge to create two child edges
        // This updates the graph of vertices
        // N.B. This is splitting the edge in half along the splitting hyperplane
        bool edge_split_performed = longest_edge->split(this->registry);

        // If 2d, then we just need to make sure vertex graph is updated correctly and then we are done
        if (is_2d()) {
            shared_ptr<SMEdge> child_edge_0 = longest_edge->child_edge_0;
            shared_ptr<SMEdge> child_edge_1 = longest_edge->child_edge_1;

            // Update the graph of vertices for the new edges and removal of the parent edge
            shared_ptr<SMVertex> longest_edge_v0 = longest_edge->v0;
            shared_ptr<SMVertex> longest_edge_v1 = longest_edge->v1;
            shared_ptr<SMVertex> longest_edge_midpoint = longest_edge->midpoint;

            longest_edge_v0->erase_bidirectional_connection(longest_edge_v1);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v0);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v1);

            return;
        }

        // Now remove the parent simplex from the mesh graph
        this->remove_simplex_from_mesh_graph(simplex);

        // longest_edge should be in the current mesh graph iff we just performed the split
        // If we just split and edge, then we need to perform a bunch of mesh graph maintainence
        if (edge_split_performed) 
        {   
            // Get the new edges
            shared_ptr<SMEdge> child_edge_0 = longest_edge->child_edge_0;
            shared_ptr<SMEdge> child_edge_1 = longest_edge->child_edge_1;

            // Assert that longest_edge is in the mesh graph.
            // assert(this->edge_to_simplex_map.contains(longest_edge));

            // Look up once and snapshot the adjacency set so that both
            // inherit_parent_edge_connections calls can share the same
            // copy (previously each call did its own .at() lookup and
            // copy of the same set).
            auto parent_it = this->edge_to_simplex_map.find(longest_edge);
            assert(parent_it != this->edge_to_simplex_map.end());
            unordered_set<shared_ptr<SMSimplex>> parent_adjacent_simplices = parent_it->second;

            // insert these new edges into the mesh graph, by inheriting connections from the parent edge
            this->inherit_parent_edge_connections(child_edge_0, parent_adjacent_simplices);
            this->inherit_parent_edge_connections(child_edge_1, parent_adjacent_simplices);

            // Remove the parent edge from the mesh graph
            this->remove_edge_from_mesh_graph(longest_edge);

            // Update the graph of vertices for the new edges and removal of the parent edge
            shared_ptr<SMVertex> longest_edge_v0 = longest_edge->v0;
            shared_ptr<SMVertex> longest_edge_v1 = longest_edge->v1;
            shared_ptr<SMVertex> longest_edge_midpoint = longest_edge->midpoint;

            longest_edge_v0->erase_bidirectional_connection(longest_edge_v1);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v0);
            longest_edge_midpoint->add_bidirectional_connection(longest_edge_v1);

            // Update non-conformity for the new child edges
            this->update_non_conformity_for_new_edge(child_edge_0, min_radius, max_depth, split_counter_threshold);
            this->update_non_conformity_for_new_edge(child_edge_1, min_radius, max_depth, split_counter_threshold);
        }

        // Add the children simplices to the mesh graph
        // Note that this should add the new edges along the splitting hyperplane
        this->add_new_simplex_to_mesh_graph_and_update_vertex_graph(
            simplex->normal_child, min_radius, max_depth, split_counter_threshold);
        this->add_new_simplex_to_mesh_graph_and_update_vertex_graph(
            simplex->opposite_child, min_radius, max_depth, split_counter_threshold);
    }

    void SMMesh::inherit_parent_edge_connections(
        const shared_ptr<SMEdge>& new_edge,
        const unordered_set<shared_ptr<SMSimplex>>& adjacent_simplices)
    {
        // UNION (not overwrite) into edge_to_simplex_map[new_edge]. new_edge may
        // already be a key in the mesh graph due to a geometric midpoint
        // coincidence: when SMEdge::split computes the midpoint of parent_edge,
        // registry.get_or_create_vertex may return an existing vertex whose
        // weight bit-exactly equals (v0+v1)/2, and registry.get_or_create_edge
        // may then return an existing SMEdge (new_edge) that was previously
        // registered as an inter-vertex edge of some other simplex in the
        // mesh graph. In that case we must preserve new_edge's pre-existing
        // adjacencies and add parent_edge's adjacencies on top of them.
        auto& dst = this->edge_to_simplex_map[new_edge];
        for (const shared_ptr<SMSimplex>& simplex : adjacent_simplices) {
            dst.insert(simplex);
            this->simplex_to_edge_map[simplex].insert(new_edge);
        }
    }

    void SMMesh::remove_edge_from_mesh_graph(shared_ptr<SMEdge> edge)
    {
        // Single lookup of edge_to_simplex_map[edge]: iterate off the
        // iterator, then erase via the iterator at the end.
        auto edge_it = this->edge_to_simplex_map.find(edge);
        if (edge_it == this->edge_to_simplex_map.end()) {
            return;
        }
        for (const shared_ptr<SMSimplex>& simplex : edge_it->second) 
        {  
            // Removes the edge from the simplex's set of edges
            auto s_it = this->simplex_to_edge_map.find(simplex);
            if (s_it != this->simplex_to_edge_map.end()) {
                s_it->second.erase(edge);
            }
        }
        this->edge_to_simplex_map.erase(edge_it);
    }

    void SMMesh::update_non_conformity_for_new_edge(
        shared_ptr<SMEdge> new_edge,
        double min_radius, 
        int max_depth, 
        int split_counter_threshold)
    {
        // Iterate directly off edge_to_simplex_map[new_edge] rather than
        // taking a copy; we do not mutate this entry in the loop body.
        const unordered_set<shared_ptr<SMSimplex>>& adjacent_simplices =
            this->edge_to_simplex_map.at(new_edge);
        for (const shared_ptr<SMSimplex>& simplex : adjacent_simplices) 
        {
            if (simplex->is_non_conforming)
            {
                continue;
            }
            if (!simplex->allowed_to_subdivide(min_radius, max_depth)
                && !this->always_allow_non_conforming_simplex_to_split) 
            {
                continue;
            }
            if (!simplex->contains_vertex(new_edge->v0) || !simplex->contains_vertex(new_edge->v1)) {
                simplex->is_non_conforming = true;
                this->non_conforming_simplices.insert(simplex);
                this->non_conforming_simplices_by_depth[simplex->depth].insert(simplex);
            }
        }
    }

    void SMMesh::remove_simplex_from_mesh_graph(shared_ptr<SMSimplex> simplex)
    {
        // Single lookup of simplex_to_edge_map[simplex]: iterate off the
        // iterator, then erase via the iterator at the end.
        auto s_it = this->simplex_to_edge_map.find(simplex);
        if (s_it != this->simplex_to_edge_map.end())
        {
            for (const shared_ptr<SMEdge>& edge : s_it->second) 
            {
                auto e_it = this->edge_to_simplex_map.find(edge);
                if (e_it != this->edge_to_simplex_map.end()) {
                    e_it->second.erase(simplex);
                }
            }
            this->simplex_to_edge_map.erase(s_it);
        }

        // Also this simplex was in the non-conforming set, remove it from
        // there. Use erase(key) which returns the count removed, avoiding
        // the extra contains() probe.
        if (this->non_conforming_simplices.erase(simplex) > 0) 
        {
            auto d_it = this->non_conforming_simplices_by_depth.find(simplex->depth);
            if (d_it != this->non_conforming_simplices_by_depth.end())
            {
                d_it->second.erase(simplex);
                if (d_it->second.empty()) 
                {
                    this->non_conforming_simplices_by_depth.erase(d_it);
                }
            }
        }
    }

    void SMMesh::add_new_simplex_to_mesh_graph_and_update_vertex_graph(
        shared_ptr<SMSimplex> simplex, 
        double min_radius, 
        int max_depth, 
        int split_counter_threshold)
    {
        // Get edges of simplex
        unordered_set<shared_ptr<SMEdge>> edges;
        for (size_t i=0; i<simplex->vertices.size(); i++) {
            for (size_t j=i+1; j<simplex->vertices.size(); j++) {
                shared_ptr<SMEdge> edge = this->registry.get_or_create_edge(simplex->vertices[i], simplex->vertices[j]);
                edges.insert(edge);
            }
        }

        // Walk every edge's partition exactly once and, for each newly
        // discovered mesh edge, perform all downstream work in the same
        // iteration: update the vertex graph, add the simplex to the
        // edge->simplex map. Previously this required three separate
        // passes over mesh_graph_edges.
        unordered_set<shared_ptr<SMEdge>> mesh_graph_edges;
        for (const shared_ptr<SMEdge>& edge : edges) {
            unordered_set<shared_ptr<SMEdge>> edge_partition = edge->get_edge_partition();
            for (const shared_ptr<SMEdge>& edge_partition_edge : edge_partition) {
                auto [mit, inserted] = mesh_graph_edges.insert(edge_partition_edge);
                if (!inserted) continue;
                edge_partition_edge->v0->add_bidirectional_connection(edge_partition_edge->v1);
                this->edge_to_simplex_map[edge_partition_edge].insert(simplex);
            }
        }

        // Update non-conformity for the new simplex
        // This new simplex is only conforming if all of its edges are in the mesh graph
        // Because mesh graph should not contain any overlapping edges, we can just compare the sizes
        if (edges.size() != mesh_graph_edges.size() 
            && (simplex->allowed_to_subdivide(min_radius, max_depth)
                || this->always_allow_non_conforming_simplex_to_split)) 
        {
            simplex->is_non_conforming = true;
            this->non_conforming_simplices.insert(simplex);
            this->non_conforming_simplices_by_depth[simplex->depth].insert(simplex);
        }

        // And add the pointers to mesh edges from this simplex. std::move
        // avoids a copy since mesh_graph_edges is not used afterwards.
        this->simplex_to_edge_map[simplex] = std::move(mesh_graph_edges);
    }
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

    // shared_ptr<SMVertex>
    size_t hash<shared_ptr<SMVertex>>::operator()(const shared_ptr<SMVertex>& v) const 
    {
        return v->hash();
    }

    size_t equal_to<shared_ptr<SMVertex>>::operator()(const shared_ptr<SMVertex>& v0, const shared_ptr<SMVertex>& v1) const 
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

    // shared_ptr<SMEdge>
    size_t hash<shared_ptr<SMEdge>>::operator()(const shared_ptr<SMEdge>& e) const 
    {
        return e->hash();
    }
    
    size_t equal_to<shared_ptr<SMEdge>>::operator()(const shared_ptr<SMEdge>& e0, const shared_ptr<SMEdge>& e1) const 
    {
        return e0->equals(*e1);
    }
}

namespace thts {
    bool operator==(const SMVertex& v0, const SMVertex& v1) 
    {
        return v0.equals(v1);
    }

    bool operator==(const shared_ptr<SMVertex>& v0, const shared_ptr<SMVertex>& v1) 
    {
        return v0->equals(*v1);
    }

    bool operator==(const SMEdge& e0, const SMEdge& e1) 
    {
        return e0.equals(e1);
    }

    bool operator==(const shared_ptr<SMEdge>& e0, const shared_ptr<SMEdge>& e1) 
    {
        return e0->equals(*e1);
    }
}