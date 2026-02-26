#pragma once

#include "thts_manager.h"

#include "mo/algorithms/simplex_maps/sm_manager.h"

#include <Eigen/Dense>

#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mo/data_structures/convex_hull.h"
#include "mo/mo_thts_types.h"



    /**
     * SimplexMap
     * 
     * Not thread safe, classes using this should protect use of this datastructure themselves

    Some notes (written in a rush and stream of consciousness, more to make sure its descriptive rather than well written or super clear) on this:

    We will start with a unit simplex, and then iteratively refine it by performing bisections
    Bisections will always be along the longest edge of the simplex
    Ties are NOT broken randomly, but by the first edge encountered

    Each bisection will create two new simplices, and so we end up with a binary tree of simplices

    Bisection using the longest edge takes care of a couple potential issues, in literature, 
    this is usually talked about some sort of smoothness or regularity of simplices. If 
    the minimum interior angle in the initial "mesh" is alpha, then iterative bisections will 
    lead to a mesh where all interior angles are at least alpha/2. This ensures that all 
    triangles are "not too thin" etc. In our case, because all interior angles (or triangular faces) 
    are 60 degrees, this leads to all possible angles being in the set {30,45,60,90,120}. And 
    means there is only a finite number of similar simplices we will make use off.

    A second consideration, which we will be considering in more detail is "conformity". 
    Two (n-d) simplices are conforming iff they share exactly a face of dimension <= n-1. 
    It's clearer in 2d with triangles, or 3d with tetrahedra. 
    In 2d, triangles are conforming iff they have no points in common, share exactly one vertex in common, or share an entire edge in common.
    In 3d, tetrahedra can be conforming if they share a triangular face in common.

    It is worth noting that with our subdivision, it is sufficient to only look for non-conformity along edges. 
    As non-conformity along a face will imply non-conformity along at least one edge.
    I think this results from the mesh being a partition of the simplex.

    Below we will talk about the binary tree of simplices and the "mesh". In the example below,
    the triangle ACD is part of the simplex map and binary tree, but is not part of the mesh.
    The mesh consists of the triangles ABC, AFE, FEC, CED.

                             A
                            /|\
                           / | \
                          / G|  \
                         /   |   \
                        /   F|____\E
                       /     |   / \
                      /      |  /   \
                     /       | /     \
                    /________|/_______\
                    B       C         D

    Typically in mesh refinement, algorithms are designed to ensure the mech is conforming. However, in our 
    applicaiton we will be refining many meshes. It will be helpful for the meshes to conform (described later).
    We will be using an "eventually-conforming" algorithm, where we keep track of edges that are 
    non-conforming and refine them gradually, WITHOUT requiring a lot of computation on every operation we perform on the simplex map.

    Each vertex corresponds to an objective weight w. And will store a value estimate for that weigting.

    Simplices will be bisected when the value estimates at the vertices of a simplex differ, indicating 
    that values within the simplex are non uniform and can benefit from refinement of the mesh.

    All edge and vertex information will be stored in hashmap objects to ensure no duplicates are created.

    So we have three objectives with this data structure:
    1. given arbitrary weight vector, quickly find the closest vertex (of a simplex in the map) to the vector
    2. maintain a graph of vertices to allow for message passing (sharing good value estimates to closeby vertices)
    3. focus the mesh refinement on most important parts of the simplex
    4. refine simplex mesh which will be "eventually conforming"

    #1:
    During the refinement process, we will produce a binary tree of simplices. 
    In the example, ABD is refined to ABC + ACD, ACD to ACE + CED, and so on.
    Each simplex will be stored in a SMSimplex object, which will keep track of the binary tree.
    Note that in the example, ACD is a simplex in the tree, but is not part of the current mesh.
    Finally, finding the "closest" vertex is a bit approximate, and also assumes that the mesh is conforming.

    #2:
    Each vertex will be stored in a SMVertex object.
    These vertices will maintain a graph of their neighbours along the current mesh edges.
    Message passing will use these edges to share value estimates.
    In the example, vertex A will currently have neighbours B, E, and F.

    #3:
    Each simplex will only only be subdivided when the value estimates at the vertices are different.
    We will thesholds for how different the value estimates need to be, and for how many "visits" to the MCTS node 
        hitting this simplex before splitting.
    This way, if the multi-objective values within a simplex are uniform, we will not unnecessarily divide it.
    Contrarily, when values are different, this is an interesting region of the simplex and we will split.

    #4:
    Refining the mesh will require keeping track of the current mesh, and the non-conforming simplices in the mesh.
    We will store a bipartite graph between edges (SMEdge objects) and simplices (SMSimplex objects) in an SMMesh 
        object.
    Each SMSimplex object (in the mesh) will have pointers to each SMEdge that is incident to it.
    Each SMEdge object will have pointers to each SMSimplex that is incident to it.
    Additionally, we will keep track of which simplices are non-conforming.
    We can tell if a simplex is non-conforming if it is adjacent to an edge which has an end vertex that is not on the 
        simplex
    In the example, ABC is non conforming, because the edge AF has an end vertex F that is not on ABC
    Note that edge AC is NOT part of the mesh, because it was split into AF and FC
    In the example we will have the following connections in the graph:
    triangle ABC -> edges { AB, BC, AF, FC }
    edge AF -> triangles { ABC, AEF }
    Finally, SMMesh will keep track of all non-conforming simplices, keeping track of the binary tree depth of the 
        simplices, so that the most "coarse" simplices can be refined first.
    
    Suppose to added the line GE to the mesh (splitting triangle AFE into AGE and GFE)
    (note that this is ignoring the rule to split along the longest edge, but is just for example)
    Then the following connections related to AF will be updated in the graph to:
    - (removed) edge AF -> triangles { ABC, AEF }
    - (added) edge AG -> triangles { ABC, AGE }
    - (added) edge GF -> triangles { ABC, GFE }
    - (removed AF, added AG, GF) triangle ABC -> edges { AB, BC, FC, AG, GF } 
    - (removed) triangle AEF -> edges { AE, EF, FA }
    - (added) triangle AGE -> edges { AG, GE, AE }
    - (added) triangle GFE -> edges { GF, FE, GE }


    Some additoinal issues, which will write up at a later date:
    - SMVertex can have exponential number of neightbours - solve with param to limit number of neighbours in message passing
    - Nearest SMVertex in lookup in SMSimplex doesn't have to be in the simplex (think point next to border of a thin 
        tall triangle and short stumpy triangle (the oppositing edge of the short stumpy tirangle can be closest)) 
        - solve with option whether to accept closest SMVertex from a SMSimplex, rather than checking all neighbouring SMSimplex's 
    - SMSimplex can also have exponential number of neighbours, this means the SMMesh graph can be exponential in size 
        - solve by providing an option whether to enforce conformity



    Longer unfinished notes on these issues:

    Some miscellaneous notes:
    - on each operation on the simplex map that may lead to a subdivision, we will check if there are any 
        non-conforming simplices
    -- if there are, we will always bisect that simplex and update the mesh (graph) accordingly

    Finally, here is a summary of the data structures that will make up the simplex map:
    - SMVertex: vertices of the mesh (containing value estimates and edges to neighbours for sharing)
    - SMSimplex: simplices used in the binary tree and mesh (contains pointers to binary tree children and if it is 
        non-conforming)
    - SMMesh: bipartite graph storing connections between SMSimplex's and SMEdge's 
    - SMEdge: edges of the mesh (only used in the SMMesh graph)

    When an MCTS node wants to use a simplex map, generally it will do the following:
    -- (optionally) sample a random vertex, for example to pick a weight vector to use for a trial
    - look up SMSimplex in the mesh, USING the binary tree to find it
    - find closest SMVertex to the current context weight
    - update value estimate at that vertex
    - share value estimates with neighbouring vertices
    -- (optionally) share value estimates to a greater radius
    -- (optionally) pick a random vertex to try push its value estimates
    - maybe subdivide the SMSimplex (and update the SMMesh / graph)
    -- if there are any non-conforming simplices, subdivide one of them (and update the SMMesh / graph)
    */


namespace thts {
    // Forward declare types (so we can define all connections before defining the class)
    struct SMVertex;
    struct SMSimplex;
    struct SMEdge;
    struct SMRegistry;
    struct SMMesh;
};


/**
 * Hash overrides
 * (think this needs to be declared before types are used in unordered_set and unordered_maps)

 Point to the class implementations
 And override hash/equals for shared_ptr versions so we can use pointers in the same way
 * 
 * Note, implemented at the end of simplex_map.cpp
*/
namespace std {
    using namespace thts;

    template<>
    struct hash<SMVertex> {
        size_t operator()(const SMVertex&) const;
    };
    template<>
    struct equal_to<SMVertex> {
        size_t operator()(const SMVertex&, const SMVertex&) const;
    };
    template<>
    bool operator==(const SMVertex& v0, const SMVertex& v1);

    template<>
    struct hash<shared_ptr<SMVertex>> {
        size_t operator()(const shared_ptr<SMVertex>&) const;
    };
    template<>
    struct equal_to<shared_ptr<SMVertex>> {
        size_t operator()(const shared_ptr<SMVertex>&, const shared_ptr<SMVertex>&) const;
    };
    template<>
    bool operator==(const shared_ptr<SMVertex>& v0, const shared_ptr<SMVertex>& v1);



    template<>
    struct hash<SMEdge> {
        size_t operator()(const SMEdge&) const;
    };

    template<>
    struct equal_to<SMEdge> {
        size_t operator()(const SMEdge&, const SMEdge&) const;
    };

    template<>
    bool operator==(const SMEdge& e0, const SMEdge& e1);

    template<>
    struct hash<shared_ptr<SMEdge>> {
        size_t operator()(const shared_ptr<SMEdge>&) const;
    };

    template<>
    struct equal_to<shared_ptr<SMEdge>> {
        size_t operator()(const shared_ptr<SMEdge>&, const shared_ptr<SMEdge>&) const;
    };

    template<>
    bool operator==(const shared_ptr<SMEdge>& e0, const shared_ptr<SMEdge>& e1);
}

namespace thts {
    

    /**
     * SMVertex
     * represents vertices of the mesh
     * Shareable value is used to mark if the value estimate can be shared with neighbours
     * - we might not want to share the value estimate if it is from an optimistic heuristic
     * - if we shared in that case, we may end up with a loop of sharing the same heuristic 
            value rather than updating the values properly from backups
    * 
    These objects need to be unique, so we will use a registry to keep track of them + construct them
    And make the constructor private so that only the registry can construct them

    Num updates refers to how many times the value estimate has been updated
    This include a "failed" update, where the value estimate did not change
    Num direct updates refers to how many times the value estimate has been updated directly, without any message passing
    */
    struct SMVertex : public std::enable_shared_from_this<SMVertex> {
        friend SMRegistry;

        Vec weight;

        int num_direct_updates;
        int num_updates;
        Vec value_estimate;
        Vec value_estimate_for_search;
        double entropy_estimate;

        std::shared_ptr<std::unordered_set<std::shared_ptr<SMVertex>>> neighbours;

    private:
        /**
         * Constructor
         */
        SMVertex(const Vec& weight, const Vec& heuristic_value_estimate, double entropy_estimate=0.0);

        /**
         * Constructor as midpoint of two other vertices
         */
        SMVertex(std::shared_ptr<SMVertex> v0, std::shared_ptr<SMVertex> v1, double ratio=0.5);


    public:
        
        /**
         * Allow vertices to be hashed and compared
         */
        size_t hash() const;
        bool equals(const SMVertex& other) const;
        bool operator==(const SMVertex& other) const;
        bool operator!=(const SMVertex& other) const;
        
        /**
         * Message passing
         Pushes value estimates to neighbours in a BFS manner
         Helper function performs the actual pushing and returns is to_vertex was updated
        */
        void share_values_message_passing(RandManager& rand_manager, int max_push_radius=1, int max_neighbours_to_push_to=-1);
    private:
        void share_values_message_passing_subset(RandManager& rand_manager, int max_neighbours_to_push_to);
        bool share_values_message_passing_helper(SMVertex& from_vertex, SMVertex& to_vertex);
    public:

        /**
         * Editing neighbourhood graph
        */
        void add_bidirectional_connection(std::shared_ptr<SMVertex> other);
        void erase_bidirectional_connection(std::shared_ptr<SMVertex> other);
    };



    /**
     * SMSimplex

     * - if we are working in D dim space (i.e. D rewards), then we are making a "D-1 simplex" using D points
     * - this D-1 simplex lies in a D-1 subspace, and the vector (1,1,...,1) is normal to this D-1 subspace

     Used for any simplex in the simplex map data structure

     For simplex:
     - dim: dimension we are working in
     - vertices: the vertices of the simplex

     For splitting:
     - split_counter: counter to keep track of how many times 'vertexes_have_different_value_estimates' has been true 
        from should_subdivide calls in a row.
     - longest_edge: a pair of vertices that are the endpoints of the longest edge of the simplex
     - split_vertex: the vertex added to split the longest edge
     - splitting_hyperplane_normal: the normal to the hyperplane that splits the simplex in two (passes through 
        the split_vertex and if longest_edge is (v0,v1) then the normal points towards v1)
    
     For binary tree:
     - depth: depth of this simplex in the binary tree
     - radius: the radius of the simplex (the maximum distance between any two vertices)
     - normal_child: the child simplex on the normal side of the splitting hyperplane
     - opposite_child: the child simplex on the opposite side of the splitting hyperplane

     For mesh:
     - is_non_conforming: if the simplex is non-conforming in the SMMesh graph

     Notes,
     the splitting hyperplane is a D-2 dimensional hyperplane
     the splitting hyperplane is defined by the D-2 following points: vertices + split_vertex - longest_edge
        (D-1 points + 1 point - 2 points = D-2 points)
     if longest_edge is (v0,v1) then we compute the normal to the hyperplane to point towards v1

     With is_non_conforming, it is worth noting that the simplex will progress through three potential states:
     - conforming (is_non_conforming is false)
     - non-conforming (is_non_conforming is true)
     - subdivided (is_non_conforming is irrelevant, as the simplex has been split, and is only part of the binary tree, 
        not the mesh graph)

     For the special case of 2D rewards, we can ignore a bunch of the geometry
     The 2D weighting, w, between the rewards is uniquely defined by the scalar value in the first dim w[0]
     In 2D weights, with w[0] varying from (left) 0 to 1 (right).
     With respect to the scalar w[0], this code will assume that the "normal" direction is always right
     */
    struct SMSimplex {
        int dim;
        std::vector<std::shared_ptr<SMVertex>> vertices;
        std::unordered_set<std::shared_ptr<SMVertex>> vertices_set;

        int split_counter;
        std::pair<std::shared_ptr<SMVertex>,std::shared_ptr<SMVertex>> longest_edge;
        std::shared_ptr<SMVertex> split_vertex; // null if leaf node
        std::shared_ptr<Vec> splitting_hyperplane_normal; // null if leaf node

        int depth;
        double radius;
        std::shared_ptr<SMSimplex> normal_child; // null if leaf node
        std::shared_ptr<SMSimplex> opposite_child; // null if leaf node

        bool is_non_conforming;

        /**
         * Constructor
         */
        SMSimplex(int dim, std::vector<std::shared_ptr<SMVertex>>& vertices, int depth);

        /**
         * Helper to check for 2D special case
         */
        bool is_2d() const;

        /**
         * Helper to check if a vertex is in this simplex
         */
        bool contains_vertex(std::shared_ptr<SMVertex> vertex) const;

        /**
         * Helper to compute a normal to a set of hyperplane points
        */
        Vec compute_hyperplane_normal(std::vector<std::shared_ptr<SMVertex>>& hyperplane_points) const;

        /**
         * Get the closest vertex to a weight from the points in this simplex
        */
        std::shared_ptr<SMVertex> get_closest_vertex(const Vec& weight) const;
        std::shared_ptr<SMVertex> operator[](const Vec& weight) const;

        /**
         * Function to create children and add them to the binary tree
         */
        void create_children(SMRegistry& registry);
        
        /**
         * return true if point is on normal side of the plane defined by halfplane_point and halfplane_normal 
            (i.e. if wegith-halfplane_point dot halfplane_normal == 0)

            halfplane_check
            if D dims, then working in D-1 dim simplex
            plane is a D-2 dim hyperplane
            the "halfplane" refers to the D-1 half plane, on the normal side of the D-2 dim plane

            more simply
            plane will be the dividing hyperplane of the children simplices
            plane_point will be a point in the plane (i.e. the split_vertex)
            plane_normal will be the normal to the plane (i.e. the splitting_hyperplane_normal)
            this checks if weight is on the normal side of the plane

         * 
         * halfplane_point:
         *      a point in the halfplane
         * halfplane_normal:
         *      the normal to the halfplane
         * weight:
         *      the weight we want to check what side of the halfplane we are on
         * 
         * Returns if 'weight' is on the normal side of the halfplane
        */
        bool halfplane_check(
            const Vec& plane_point, 
            const Vec& plane_normal, 
            const Vec& weight) const;

        /**
         * Assuming weight is inside this simplex, return the child simplex that contains it
         */
        std::shared_ptr<SMSimplex> traverse(const Vec& weight) const;

        /**
         * If this node is a leaf in the binary tree
         */
        bool is_leaf() const;
        
        /**
         * Checks if simplex is worth subdividing

         Never subdivide if radius is less than min_radius
         Will return false if already subdivided
         Wont subdivide if depth is greater than max_depth

         After these checks,
         Should subdivide will check if vertexes_have_different_value_estimates is true, 
         and increment split counter
         if split counter is greater than threshold, then we should subdivide

         does not actually call 'create_children'

         */
        bool vertexes_contain_multiple_unique_values() const;
        bool allowed_to_subdivide(double min_radius, int max_depth) const;
        bool should_subdivide(double min_radius, int max_depth, int split_counter_threshold) const;
    }

    /**
     * SMEdge

     Data structure for edges of simplices
     Main point is to facilitate the mesh bipartite graph to identify non-conforming simplices
     And use to link vertexes in their graph

     This is basically an unordered pair of SMVertex pointers

     We will also keep track of edges that formed from splitting this edge (in another binary tree)
     So that when we need to refine the mesh, we can use this edge to lookup the relevant edges that are currently in 
        the mesh graph

    These objects need to be unique, so we will use a registry to keep track of them + construct them
    And make the constructor private so that only the registry can construct them
     */
    struct SMEdge {
        friend SMRegistry;

        std::shared_ptr<SMVertex> v0;
        std::shared_ptr<SMVertex> v1;
        std::shared_ptr<SMVertex> midpoint;
        std::shared_ptr<SMEdge> child_edge_0;
        std::shared_ptr<SMEdge> child_edge_1;
    
    private:
        // Private constructore to force use of registry constructor
        SMEdge(std::shared_ptr<SMVertex> v0, std::shared_ptr<SMVertex> v1);
    
    public:
        // Allow edge to be hashed and compared
        size_t hash() const;
        bool equals(const SMEdge& other) const;
        bool operator==(const SMEdge& other) const;
        bool operator!=(const SMEdge& other) const;

        // Split this edge
        void split(SMRegistry& registry);

        // Get the set of smallest edges that partition this edge
        std::shared_ptr<std::unordered_set<std::shared_ptr<SMEdge>>> get_edge_partition() const;
        void get_edge_partition_helper(std::shared_ptr<SMEdge> edge, std::unordered_set<std::shared_ptr<SMEdge>>& partition) const;

        // Find closest point on the edge to a given point
        Vec find_closest_point_on_edge(const Vec& point) const;
        double find_closest_point_on_edge_ratio(const Vec& point) const;
    };

    /**
    SMRegistry

    Used to keep track of data structures that need to be unique and it is a non-trivial task to ensure this
     */
    struct SMRegistry {
        
        std::unordered_map<std::shared_ptr<SMVertex>, std::shared_ptr<SMVertex>> vertex_map;
        std::unordered_map<std::shared_ptr<SMEdge>, std::shared_ptr<SMEdge>> edge_map;

        SMRegistry() = default;
        virtual ~SMRegistry() = default;

        // Public interface for SMVertex construction
        std::shared_ptr<SMVertex> get_or_create_vertex(const Vec& weight, const Vec& value_estimate, double entropy_estimate=0.0);
        std::shared_ptr<SMVertex> get_or_create_vertex(std::shared_ptr<SMVertex> v0, std::shared_ptr<SMVertex> v1, double ratio=0.5);

        // Public interface for SMEdge construction
        std::shared_ptr<SMEdge> get_or_create_edge(std::shared_ptr<SMVertex> v0, std::shared_ptr<SMVertex> v1);

    private:
        // Helper functions to lookup the unique version of a vertex or edge
        std::shared_ptr<SMVertex> lookup_unique_vertex(std::shared_ptr<SMVertex> vertex);
        std::shared_ptr<SMEdge> lookup_unique_edge(std::shared_ptr<SMEdge> edge);
    };

    /**
    SMMesh

    In general, this class is in charge of maintaining all of the geometry of the simplex map.
    Explicitly, it is in charge of orchestrating the creation and maintenance of SMSimplex, SMEdge and SMVertex objects

    Keeps track of the mesh of simplices
    And maintains the graph of simplices along the edges of the simplices

    Maintains a bipartite graph of simplices and edges to maintain the mesh
    Keeps track of all simplices and edges currently forming the mesh
    And keeps track of all non-conforming simplices, from what depth of the binary tree they are at

    A bit more specifically, we are keeping track of simplices with the following properties:
    - they are non-conforming
    - they meet all conditions (apart from non having uniform value estimates) to be able to be refined
    - they are part of the mesh graph

    Note that a simplex is part of the mesh graph iff it is a leaf in the binary tree

    Additionally, 2D rewards will be an edge case. 
    This is because all simplices are 1D line segments, which makes the use of SMEdge unnecessary
    Moreover, 1D simplices will always be conforming
    Hence, when rewards are 2D, we will ignore all mesh graph logic and just use the binary tree 

    */
    struct SMMesh {
        int dim;
        bool find_exact_closest_vertex;
        bool eventually_conforming_mesh;
        bool always_allow_non_conforming_simplex_to_split;
        SMRegistry registry;
        std::shared_ptr<SMSimplex> root_simplex; // binary tree of simplices
        std::unordered_set<std::shared_ptr<SMVertex>> all_vertices_set;
        std::vector<std::shared_ptr<SMVertex>> all_vertices_vector;

        // Mesh graph variables
        std::unordered_map<std::shared_ptr<SMSimplex>, std::unordered_set<std::shared_ptr<SMEdge>>> simplex_to_edge_map;
        std::unordered_map<std::shared_ptr<SMEdge>, std::unordered_set<std::shared_ptr<SMSimplex>>> edge_to_simplex_map;
        std::unordered_set<std::shared_ptr<SMSimplex>> non_conforming_simplices;
        std::map<int,std::queue<std::shared_ptr<SMSimplex>>> non_conforming_simplices_by_depth;

        // Constructore
        SMMesh(int dim, bool find_exact_closest_vertex=true, bool eventually_conforming_mesh=true, bool always_allow_non_conforming_simplex_to_split=true);

        // Desstructor
        // Needs to make sure that the SMVertex graph gets cleaned up (circular references of shared_ptr could lead to 
        // memory leaks)
        virtual ~SMMesh();

        // Initialise the mesh with a root simplex
        void initialise_mesh(Vec& heuristic_value_estimate);

        // Sample a random vertex from the mesh
        std::shared_ptr<SMVertex> sample_random_vertex(RandManager& rand_manager) const;

        // Get the (smallest) simplex (leaf node in binary tree) containing a given weight
        std::shared_ptr<SMSimplex> get_simplex(const Vec& weight) const;

        // Get the closest vertex to a given weight (from the simplex containing the weight)
        std::shared_ptr<SMVertex> get_closest_vertex(const Vec& weight) const;
        std::shared_ptr<SMVertex> get_closest_vertex(const Vec& weight, std::shared_ptr<SMSimplex> simplex) const;

        // Reading from a vertex
        int get_num_updates(std::shared_ptr<SMVertex> vertex) const;
        Vec get_value_estimate(std::shared_ptr<SMVertex> vertex) const;
        Vec get_value_estimate_for_search(std::shared_ptr<SMVertex> vertex) const;
        double get_entropy_estimate(std::shared_ptr<SMVertex> vertex) const;

        // Update a value estimate for a vertex
        void update_vertex_values_and_share(
            RandManager& rand_manager,
            std::shared_ptr<SMVertex> vertex, 
            int max_push_radius,
            int max_neighbours_to_push_to
            const Vec& value_estimate, 
            const Vec& value_estimate_for_search, 
            double entropy_estimate=0.0,);

        // Maybe subdivide a simplex, if it meets the conditions to warrent it
        // Additionally, if there are any non-conforming simplices, the lowest depth one will be subdivided
        void maybe_subdivide(
            std::shared_ptr<SMSimplex> simplex, 
            double min_radius, 
            int max_depth, 
            int split_counter_threshold);

        // Pretty print the mesh
        std::string get_pretty_print_string() const;

        // Return a convex hull approximation
        ConvexHull get_approximate_convex_hull() const;

        // We will keep helper functions private to keep a clean interface
    private:
        // Sometimes 2D reward will be an edge case
        bool is_2d() const;

        // Get the lowest depth non-conforming simplex
        std::shared_ptr<SMSimplex> pop_lowest_depth_non_conforming_simplex();

        // Helper to orchestrate the subdivision of a simplex
        // Add children to binary tree
        // Adds children to mesh graph
        // Removes parent from mesh graph
        // Adds split edges to mesh graph
        // Updates non-conforming simplices
        void subdivide_simplex(
            std::shared_ptr<SMSimplex> simplex, 
            double min_radius, 
            int max_depth, 
            int split_counter_threshold);

        // Helper to maybe add a new edge to the graph, and inherit connections (to simplices) from the parent edge
        void inherit_parent_edge_connections(std::shared_ptr<SMEdge> new_edge, std::shared_ptr<SMEdge> parent_edge);

        // Helper to update simplices that may now be non-conforming, checking simplices that are adjacent to the edge
        void update_non_conformity_for_new_edge(
            std::shared_ptr<SMEdge> new_edge
            double min_radius, 
            int max_depth, 
            int split_counter_threshold);

        // Helper to remove an edge from the mesh graph
        void remove_edge_from_mesh_graph(std::shared_ptr<SMEdge> edge);

        // Helper to remove simplex from the mesh graph
        void remove_simplex_from_mesh_graph(std::shared_ptr<SMSimplex> simplex);

        // Get the set of SMEdges that are adjacent to a given simplex
        std::unordered_set<std::shared_ptr<SMEdge>> get_edges_adjacent_to_simplex(std::shared_ptr<SMSimplex> simplex) const;

        // Helper to add new simplices to the mesh graph
        void add_new_simplex_to_mesh_graph(
            std::shared_ptr<SMSimplex> simplex, 
            double min_radius, 
            int max_depth, 
            int split_counter_threshold);
    };


    /**
    So, as I was implementing it, it turned out that SMMesh was basically the SimplexMap
    Maybe I could have seperated the logic somehow, but SimplexMap would have been a very thin wrapper around SMMesh
    I still want to use "SimplexMap" as the name for the class in algorithms.
    But I am also attached to SMMesh, because that class does maintain the mesh.
    So, I will typedef :)
    */
    typedef SMMesh SimplexMap;
}

