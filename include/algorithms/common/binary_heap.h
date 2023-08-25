#include <cassert>
#include <vector>
#include <unordered_map>

namespace thts {

    /**
     * An implementation of a key, value binary heap
    */
    template <typename K>
    class BinaryHeap {
        
        private:
            struct BinHeapItem {
                K key;
                double value;
            };

        protected:
            std::vector<BinHeapItem<K>> heap;
            std::unordered_map<K,int> heap_indices;

        public:
            BinaryHeap() : heap(), heap_indices() {};

            BinaryHeap(int max_elements) : heap(), heap_indices() {
                heap.reserve(max_elements);
                heap_indices.reserve(max_elements);
            };

            /**
             * Fills the heap with some initial value and then sorts it to satisfy the max heap property in O(n) time
            */
            void fill_and_heapify(std::unordered_map<K,double>& init_values) {
                assert(heap.size() > 0);
                int i = 0;
                for (std::pair<K,double> pr : init_values) {
                    heap[i].key = pr.first;
                    heap[i].value = pr.second;
                    heap_indices[pr.first] = i++;
                }
                heapify();
            };

            K peek_top_key() {
                return heap[0].key;
            };

            double peek_top_value() {
                return heap[0].value;
            };

            /**
             * Updates a value in the heap, and inserts it if it doesn't exist
             * Insert by placing at the end of the arrays and sifting up
            */
            void update_or_insert(K key, double value) {
                // Update value if can find it, otherwise insert if doesn't exist
                if (heap_indices.find(key) != heap_indices.end()) {
                    heap[heap_indices[key]].value = value;
                } else {
                    int new_indx = heap.size();
                    heap.push_back(BinHeapItem<K>());
                    heap[new_indx].key = key;
                    heap[new_indx].value = value;
                    heap_indices[key] = new_indx;
                }

                // Sift up or down into correct place
                int i = heap_indices[key];
                int p = parent(indx);
                if (heap[p].value < heap[i].value) {
                    sift_up(i);
                } else {
                    sift_down(i);
                }
            };

        private:
            inline int parent(int i) {
                return (i - 1) / 2;
            };

            inline int left_child(int i) {
                return 2*i + 1;
            };
            
            inline int right_child(int i) {
                return 2*i + 2;
            };

            /**
             * Swapping two items in the heap via their indices
            */
            void swap(int i, int j) {
                K key_i = heap[i].key;
                K key_j = heap[j].key;
                double val_i = heap[i].value;
                double val_j = heap[j].value;

                heap_indices[key_i] = j;
                heap_indices[key_j] = i;
                heap[i].key = key_j;
                heap[i].key = val_j;
                heap[j].key = key_i;
                heap[j].key = val_i;
            };

            /**
             * Assumes that heap properties are satisfied everywhere exept at the given index
             * And that the value needs to be moved up
            */
            void sift_up(int i) {
                int p = parent(i);
                while (i > 0 && heap[p].value < heap[i].value) {
                    swap(i,p);
                    i = p;
                    p = parent(i);
                }
            };

            /**
             * Assumes that 'i+1' to 'heap.size()-1' satisfies max heap property
             * This function sifts 'i' down into the heap
             * On each loop it checks for the largest out of the current index and its children
             * If the current index is largest, break, we satisfy max heap
             * Otherwise swap and repeat
            */
            void sift_down(int i) {
                while (true) {
                    int left = left_child(i);
                    int right = right_child(i);
                    int largest = i;
                    if (left < heap.size() && heap[left].value > heap[largest].value) {
                        largest = left;
                    }
                    if (right < heap.size() && heap[right].value > heap[largest].value) { 
                        largest = right;
                    }
                    if (largest == i) {
                        break;
                    }
                    swap(i, largest);
                    i = largest;
                }
            };

            /**
             * Assumes that 'heap' and 'heap_indices' are filled, but dont necessaruly satisfy the max heap property
            */
            void heapify() {
                for (int i=heap.size()-1; i>=0; i--) {
                    sift_down(i);
                }
            };
    };
}