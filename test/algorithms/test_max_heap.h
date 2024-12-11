#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "algorithms/common/max_heap.h"

#include <iostream>


namespace thts::test {
    using namespace std;
    using namespace thts;

    /**
     * Mocker for ThtsEnv
     * Adds ability to index the heap directly
     */
    template <typename K>
    class MockMaxHeap : public MaxHeap<K> {
        public:
            MockMaxHeap() : MaxHeap<K>() {};
            MockMaxHeap(int max_elements) : MaxHeap<K>(max_elements) {};

            K peek_key(int index) {
                return thts::MaxHeap<K>::heap[index].key;
            };

            double peek_value(int index) {
                return thts::MaxHeap<K>::heap[index].value;
            };

            void print_heap() {
                cout << "{" << endl;
                for (unsigned int i=0; i<thts::MaxHeap<K>::heap.size(); i++) {
                    cout << thts::MaxHeap<K>::heap[i].key << ", " << thts::MaxHeap<K>::heap[i].value << endl;
                }
                cout << "}" << endl;
            };
    };
}