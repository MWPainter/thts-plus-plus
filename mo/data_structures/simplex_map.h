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


// Forward declare namespace
namespace thts {
    // Forward declare types
    struct NGV;
    struct LSE;

    /**
     * Helper typedef
     * - points out that the pair will be hashed in an unordered way
    */
    typedef std::pair<std::shared_ptr<NGV>,std::shared_ptr<NGV>> UnorderedNGVPair;
};

/**
 * Hash overrides
 * (think this needs to be declared before types are used in unordered_set and unordered_maps)
 * 
 * TODO long term: put PF + CH and all this simplex map stuff in a mo/structs folder, and split this file up into 
 *      something like simplex_map_types.h and simplex_map.h. Theres just a lot of classes defined here
 * 
 * Note, implemented at the end of simplex_map.cpp
*/
namespace std {
    using namespace thts;

    template<>
    struct hash<shared_ptr<NGV>> {
        size_t operator()(const shared_ptr<NGV>&) const;
    };
    template<>
    struct equal_to<shared_ptr<NGV>> {
        size_t operator()(const shared_ptr<NGV>&, const shared_ptr<NGV>&) const;
    };
    template<>
    bool operator==(const shared_ptr<NGV>& v0, const shared_ptr<NGV>& v1);

    template<> 
    struct hash<shared_ptr<LSE>> {
        size_t operator()(const shared_ptr<LSE>&) const;
    };
    template<> 
    struct equal_to<shared_ptr<LSE>> {
        size_t operator()(const shared_ptr<LSE>&, const shared_ptr<LSE>&) const;
    };

    template<>
    struct hash<UnorderedNGVPair> {
        size_t operator()(const UnorderedNGVPair&) const;
    };
    template<>
    struct equal_to<UnorderedNGVPair> {
        size_t operator()(const UnorderedNGVPair&, const UnorderedNGVPair&) const;
    };
}


namespace thts {
    // Forward declare mcts classes using these classes
    class SmThtsCNode;
    class SmThtsDNode;
    class SmBtsCNode;
    class SmBtsDNode;

    // Forward declare simplex map so NGV and LSE can use it
    class SimplexMap;


    /**
     * Neighbourhood Graph Vertex
     * 
     * TODO long term: make the class templated with a data type. Then can have a value estimate payload and entropy 
     *      playlist
     * TODO long term: make these structs private data types inside the SimplexMap class
     * 
     * The need for value_esimtate_from_backup comes from not wanting to share heuristic values. Consider a case for 
     * example where the heuristic is the zero vector [0,0], and all of your rewards are negative. So the values are 
     * [-a -b], for some a,b >= 0. In this case, the message passing will keep the heuristic values always, rather than 
     * the more accurate dp estimates. So we mark if the value estimate is from a backup, so we can avoid pulling 
     * innacurate heuristic values.
     * 
     * Args:
     *      value_estimate: Best value estimate for this weight/context
     *      entropy: Entropy estimate
     *      pure_backup_value_estimate: If the value estimate has been computed purely from backup values (rather than heuristic)
     *      weight: The weight/context for this node/vertex
     *      neighbours: A set of neighbour NGV vertices
    */
    struct NGV : public std::enable_shared_from_this<NGV> {
        Eigen::ArrayXd value_estimate;
        double entropy;
        bool pure_backup_value_estimate;

        Eigen::ArrayXd weight;
        std::shared_ptr<std::unordered_set<std::shared_ptr<NGV>>> neighbours;

        NGV(Eigen::ArrayXd weight, Eigen::ArrayXd init_val_estimate, double init_entr_estimate);
        NGV(NGV& v0, NGV& v1, double ratio);

        size_t hash() const;
        bool equals(const NGV& other) const;
        bool operator==(const NGV& other) const;
        bool operator!=(const NGV& other) const;
        
        /**
         * Message passing
        */
        void share_values_message_passing();
        void share_values_message_passing_push();
        void share_values_message_passing_pull();
        void share_values_message_passing_helper_push(NGV& other);
        void share_values_message_passing_helper_pull(NGV& other);

        /**
         * Editing neighbourhood graph
        */
        void add_connection(std::shared_ptr<NGV> other);
        void erase_connection(std::shared_ptr<NGV> other);

        /**
         * Get the contextual value of weighting 'ctx' from this node (or with ctx=weight)
        */
        double contextual_value_estimate();
        double contextual_value_estimate(Eigen::ArrayXd& ctx);
    };

    /**
     * (Long) Simplex Edge
     * 
     * Represents an edge of a simplex
     * If the edge of a simplex coincides with a larger edge from a parent edge, we *dont* make a new one
     * 
     * Args:
     *      v0: 
     *          One NGV end of the edge
     *      v1: 
     *          One NGV end of the edge
     *      ratios: 
     *          A map from NGV to ratios. If r = ratios[ngv] then ngv = r * v0 + (1-r) * v1
     *      interpolated_vertex_tree:
     *          A tree of interpolated vertices
    */
    struct LSE : public std::enable_shared_from_this<LSE> {
        std::shared_ptr<NGV> v0;
        std::shared_ptr<NGV> v1;
        std::unordered_map<std::shared_ptr<NGV>,double> ratios;
        std::map<double,std::shared_ptr<NGV>> interpolated_vertices;

        LSE(std::shared_ptr<NGV> v0, std::shared_ptr<NGV> v1);

        size_t hash() const;
        bool equals(const LSE& other) const;
        bool operator==(const LSE& other) const;
        bool operator!=(const LSE& other) const;

        /**
         * Returns the ngv with maximum ratio r such that r<'ratio'
         * I.e. the node left of a "node" at 'ratio' (n.b. the "node" doesnt have to exist) 
         */
        std::map<double,std::shared_ptr<NGV>>::iterator left_vertex(double ratio);

        /**
         * Returns the ngv with minimum ratio r such that r>'ratio'
         * I.e. the node left of a "node" at 'ratio' (n.b. the "node" doesnt have to exist) 
         */
        std::map<double,std::shared_ptr<NGV>>::iterator right_vertex(double ratio);

        /**
         * Get the closest NGV lying on this edge 
         */
        std::shared_ptr<NGV> get_closest_ngv(const Eigen::ArrayXd& w);

        /**
         * Inserts an NGV between two NGVs that are on this edge
         * 
         * Assumes the x0 and x1 already exist on this edge
         * v is an NGV that has already been created
         * v = r * x0 + (1-r) * x1
         * 
         * Important: updates the neighbours of vertices in the neighbourhood graph for the insertion
        */
        void insert(
            std::shared_ptr<NGV> v, 
            std::shared_ptr<NGV> x0, 
            std::shared_ptr<NGV> x1, 
            double r, 
            SimplexMap& simplex_map);
    };

    /**
     * Triangulation
     * 
     * A struct containing the data for a triangulation of a simplex
     * 
     * Constructor reads in the data from the precomputed triangulation computed from the python script 
     * 'compute_triangulations.py'
     * 
     * Args:
     *      d: 
     *          The number of vertices in a simplex (and the number of rewards were working with)
     *      e: 
     *          The number of edges (e=0.5*d*(d-1)) in the simplex (and the number of vertices that will be added on 
     *          the edges)
     *      edge_points:
     *          A list of (i,j,r) tuples, where 0<=i,j<d are indices into the vertex points, and r specifies to add a 
     *          point at r*i + (1-r)*j
     *      simplices:
     *          A list of lists of points. Each list of points forms a simplex in the triangulation
    */
    struct Triangulation {
        int d;
        int e;
        std::vector<std::tuple<int,int,double>> edge_points;
        std::vector<std::vector<int>> simplices;

        Triangulation(int dim);
    };

    /**
     * Tree Node
     * - each tree node is synonomous with a simplex
     * - if we are working in D dim space (i.e. D rewards), then we are making a "D-1 simplex" using D points
     * - this D-1 simplex lies in a D-1 subspace, and the vector (1,1,...,1) is normal to this D-1 subspace
     * 
     * TODO: renamde this to a simplex node (SN)
     * 
     * Args:
     *      dim:
     *          The dimension we are working in (dimension of the reward)
     *      depth: 
     *          The depth of this node in the simplex map tree
     *      centroid: 
     *          The centroid of this simplex
     *      l_inf_norm:
     *          The infi{nity norm of this simplex. This infinity norm is given by: 
     *              max_{w0,w1 \in simplex_vertices} ||w0-w1||_inf
     *      split_counter:
     *          A counter that keeps track of how many times 'maybe_subdivide' is called 
     *      simplex_vertices: 
     *          The NGV's that form the simplex 
     *      hyperplane_normals:
     *          hyperplane_normals[v] is the normal to the hyperplane containing the points simplex_vertices - {v} 
     *      children: 
     *          Child nodes
     * 
     * If sm_manager.use_triangulation == false, then we build a binary tree:
     *      splitting_edge_normal_side_vertex:
     *          The vertex at the end of the splitting edge which is the side that the normal vertex points
     *      splitting_edge_opposite_side_vertex:
     *          The vertex at the end of the splitting edge which is oppositve the side that the normal vertex points
     *      splitting_edge_new_vertex:
     *          The new vertex added to split this simplex into two child simplices
     *      splitting_hyperplane_normal:
     *          The normal to the hyperplane that splits the simplex in two
     *      normal_side_child:
     *          The TN child on the normal side of the splitting hyperplane
     *      opposite_side_child:
     *          The TN child on the opposite side of the splitting hyperplane
     * 
     * Some notes when revising this. 
     * 'splitting_edge' refers to the edge of the simplex that is split by the two children of this node
     * 'splitting_hyperplane_normal' is the normal to the seperating hyperplane between the two children
     * 'splitting_edge_normal_side_vertex' and 'splitting_edge_opposite_side_vertex' are the vertices at the ends of 
     *      the 'splitting_edge'
     * 'splitting_edge_new_vertex' is the new NGV created as a result of splitting this simplex
    */
    struct TN {
        SimplexMap& simplex_map;
        int dim;
        int depth;
        Eigen::ArrayXd centroid;
        double l_inf_norm;
        int split_counter;
        mutable std::shared_ptr<std::vector<std::shared_ptr<NGV>>> simplex_vertices;
        std::shared_ptr<std::unordered_map<std::shared_ptr<NGV>,Eigen::ArrayXd>> hyperplane_normals;
        std::shared_ptr<std::unordered_set<std::shared_ptr<TN>>> children;

        std::shared_ptr<NGV> splitting_edge_normal_side_vertex;
        std::shared_ptr<NGV> splitting_edge_opposite_side_vertex;
        std::shared_ptr<NGV> splitting_edge_new_vertex;
        Eigen::ArrayXd splitting_hyperplane_normal;
        std::shared_ptr<TN> normal_side_child;
        std::shared_ptr<TN> opposite_side_child;

        TN(SimplexMap& simplex_map, int dim, int depth, std::shared_ptr<std::vector<std::shared_ptr<NGV>>> simplex_vertices);

        /**
         * On construction want to ensure that all of the simplex_vertices are (fully) connected 
        */
        void _ensure_neighbourhood_graph_connected();

        /**
         * Helper to compute a normal to a set of hyperplane points
        */
        Eigen::ArrayXd compute_hyperplane_normal(std::vector<std::shared_ptr<NGV>>& hyperplane_points) const;

        /**
         * Lazy init hyperplane normals
         * (Lazy part of construction)
        */
        void lazy_compute_hyperplane_normals() const;

        // size_t hash() const;
        // bool equals(const NGV& other) const;
        // bool operator==(const NGV& other) const;
        // bool operator!=(const NGV& other) const;

        /**
         * If this node has any children
        */
        bool has_children() const;

        /**
         * Create child TN's using the triangulation
        */
        void create_children(SmThtsManager& sm_manager);
        void create_children_binary_tree();
        void create_children_triangulation(SmThtsManager& sm_manager);

        /**
         * Get child
         * 
         * Returns a child node (simplex) that contains 'weight'
        */
        std::shared_ptr<TN> get_child(const Eigen::ArrayXd& weight) const;
        std::shared_ptr<TN> get_child_binary_tree(const Eigen::ArrayXd& weight) const;
        std::shared_ptr<TN> get_child_triangulation(const Eigen::ArrayXd& weight) const;

        /**
         * Returns true if 'weight' is contained in this simplex
         * 
         * 'weight' is inside the (D-1) simplex iff it is the same side of the (D-2) plane defined by each of the D 
         * many (D-2) face's as 'centroid'
        */
        // bool contains_weight(const Eigen::ArrayXd& weight) const;
        bool contains_weight(const Eigen::ArrayXd& weight, bool debug=false) const;
        bool contains_weight_2d(const Eigen::ArrayXd& weight) const;

        /**
         * Returns if 'weight' is the same side of the hyperplane (defined by a point in the halfplane and the normal 
         * to the halfplane) as the 'centroid' of the simplex
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
            const Eigen::ArrayXd& halfplane_point, 
            const Eigen::ArrayXd& halfplane_normal, 
            const Eigen::ArrayXd& weight) const;

        /**
         * Get the closest vertex in the neighbourhood graph
        */
        std::shared_ptr<NGV> get_closest_ngv_vertex(const Eigen::ArrayXd& ctx) const;
        std::shared_ptr<NGV> operator[](const Eigen::ArrayXd& ctx) const;

        /**
         * Getting value from this TN using simplex vertices
         * If we have children, then we would get a better estimate by using this function in the child where 
         * 'child.contains(ctx)' is true
        */
        Eigen::ArrayXd get_best_value_estimate(const Eigen::ArrayXd& ctx) const;

        /**
         * Maybe split node
         * Needs information from the simplex map and a triangulation object to do so though
         * This can come from the thts node and manager object respectively though
         * 
         * l_inf_thresh is the threshold of l_inf_norm below which we stop bothering to make any more children
         * visit_thresh is the threshold for how many times we need to visit this node (with any of or adjactent NGV 
         *      value_estimate's being different) to allow for a split
         * max_depth is the threshold for the maximum depth of the TN tree
        */
        void maybe_subdivide(SmThtsManager& sm_manager);
    };

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
    class SimplexMap {
        friend SmThtsCNode;
        friend SmThtsDNode;
        friend SmBtsCNode;
        friend SmBtsDNode;

        friend NGV;
        friend LSE;
        friend TN;

        // TODO: change this back to protected after debugging stuff
        // protected:
        public:
            int dim;
            std::shared_ptr<TN> root_node;
            std::shared_ptr<std::vector<std::shared_ptr<NGV>>> n_graph_vertices;
            std::shared_ptr<std::unordered_set<std::shared_ptr<NGV>>> n_graph_vertex_set;
            std::unordered_map<UnorderedNGVPair,std::shared_ptr<LSE>> lse_map;

        public:
            /**
             * Constructor
             * 
             * Default val is the initial value to use for each Neighbourhood Graph Vertex
            */
            SimplexMap(int reward_dim, Eigen::ArrayXd default_val);

            /**
             * Destructor
             * 
             * The graph of NGV's will have pointers to each other that will keep ref counts > 0 and never get freed, 
             * unless we explicitly clean it up.
             */
            ~SimplexMap();

            /**
             * Gets the LSE that v0 and v1 lie on (from 'lse_map')
             * If an LSE for these vertices doesnt exist, it makes it and adds to 'lse_map' 
            */
            std::shared_ptr<LSE> get_or_create_lse(std::shared_ptr<NGV> v0, std::shared_ptr<NGV> v1);

            /**
             * Register a pair of NGVs with an LSE in the lse_map
            */
            void register_vertices_with_lse(
                std::shared_ptr<NGV> v0, std::shared_ptr<NGV> v1, std::shared_ptr<LSE> edge);

            /**
             * Get the leaf TN node containing a given context
            */
            std::shared_ptr<TN> get_leaf_tn_node(const Eigen::ArrayXd& ctx) const;
            std::shared_ptr<TN> operator[](const Eigen::ArrayXd& ctx) const;

            /**
             * Sample a random weight (vertex) from the neighbourhood graph
            */
            std::shared_ptr<NGV> sample_random_ngv_vertex(RandManager& rand_manager) const;

            /**
             * Prett print
            */
            std::string get_pretty_print_string() const;

            /**
             * Return a convex hull approximation
             */
            ConvexHull get_approximate_convex_hull() const;
    };
}