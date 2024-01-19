#pragma once

#include <Eigen/Dense>

#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <unordered_set>



namespace thts {
    class SmtThtsCNode;
    class SmtThtsDNode;
    class SmtBtsCNode;
    class SmtBtsDNode;

    /**
     * Neighbourhood Graph Vertex
    */
    struct NGV {
        // TODO: Move value estimate into subclass to allow for other things to be contained in node 
        // TODO: actually make this templated, 
        // The map interface is 
        Eigen::ArrayXd value_estimate;

        Eigen::ArrayXd point;
        std::shared_ptr<std::unordered_set<std::shared_ptr<NGV>>> edges;

        size_t hash() const;
        bool equals(const NGV& other) const;
        bool operator==(const NGV& other) const;
        bool operator!=(const NGV& other) const;
        // todotodo override std hash
        
        /**
         * Message passing
        */
        void share_values_message_passing();
        void share_values_message_passing_helper(NGV& other);

        NGV(Eigen::ArrayXd point, Eigen::ArrayXd init_val_estimate);
    };

    // /**
    //  * Neighbourhood Graph Edge
    //  * (undirected)
    // */
    // struct NGE {
    //     std::shared_ptr<NGV> v0;
    //     std::shared_ptr<NGV> v1;

    //     size_t hash() const
    //     {
    //         return thts::helper::hash_combine(v0->hash(),v1->hash());
    //     };
    //     bool equals(const NGE& other) const 
    //     {
    //         return ((v0->equals(*other.v0) && v0->equals(*other.v0))
    //                 || (v0->equals(*other.v1) && v1->equals(*other.v0))) 
    //     };
    //     bool operator==(const NGE& other) const {
    //         return equals(other);
    //     };
    //     // todotodo override std hash

    //     NGE(std::shared_ptr<NGV> v0, std::shared_ptr<NGV> v1) : v0(v0), v1(v1)
    //     {
    //     };
    // };

    /**
     * Tree Node
    */
    struct TN {
        int depth;
        Eigen::ArrayXd point;
        std::shared_ptr<TN> psuedo_child; // child with the psuedo_child.point == point (for recursing into this point)
        std::shared_ptr<std::unordered_set<std::shared_ptr<TN>>> children;
        std::shared_ptr<std::unordered_set<std::shared_ptr<NGV>>> simplex_neighbourhood;

        bool has_children();

        size_t hash() const;
        bool equals(const NGV& other) const;
        bool operator==(const NGV& other) const;
        bool operator!=(const NGV& other) const;
        // todotodo override std hash

        /**
         * Getting value from this TN using simplex neighbourhood
        */
        Eigen::ArrayXd get_value_estimate(const Eigen::ArrayXd& ctx) const;

        /**
         * TN traversing
        */
        std::shared_ptr<TN> traverse_one_step(const Eigen::ArrayXd& ctx);
        bool other_tn_closer_to_ctx(Eigen::ArrayXd& ctx, TN& other);

        TN(int depth, Eigen::ArrayXd point);
    }

    class SimplexMap {
        friend SmtThtsCNode;
        friend SmtThtsDNode;
        friend SmtBtsCNode;
        friend SmtBtsDNode;

        protected:
            Eigen::ArrayXd default_val;
            std::shared_ptr<TN> root_node;
            std::shared_ptr<std::vector<std::shared_ptr<TN>>> tree_nodes;
            std::shared_ptr<std::unordered_set<std::shared_ptr<NGV>>> n_graph_vertices;
            // std::unordered_set<std::shared_ptr<NGE>> n_graph_edges;
            std::mutex lock;

        public:

            SimplexMap(int reward_dim, Eigen::ArrayXd default_val);

            /**
             * Lookup closest tree node
            */
            std::shared_ptr<TN> operator[](const Eigen::ArrayXd& ctx) const;
            std::shared_ptr<TN> lookup_node_closest_to_context(const Eigen::ArrayXd& ctx) const;

            /**
             * Splitting
            */
            void split_at(std::shared_ptr<TN> tn_ptr);

            /**
             * Sample a random node
            */
            std::shared_ptr<TN> get_random_node(RandManager& rand_manager);

            /**
             * Prett print
            */
            std::string get_pretty_print_string();
    };
}





/**
 * Hash overrides
*/
namespace std {
    using namespace thts;

    template<> 
    struct hash<shared_ptr<NGV>> {
        size_t operator()(const shared_ptr<NGV>&) const;
    };

    template<> 
    struct hash<shared_ptr<TN>> {
        size_t operator()(const shared_ptr<TN>&) const;
    };
}