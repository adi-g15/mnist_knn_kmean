#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <span>
#include <type_traits>
#include <vector>

using std::vector;

// Currently only for vector, or span
// template<typename T, ReversibleContainer ContainerType>
template <typename ContainerType> class Subset {
    using T = typename ContainerType::value_type;

    // Idea from the "Competetive programmer handbook", we can use a binary
    // number to denote a subset, with 1 for indices of elements included in
    // this 'subset' and 0 for indices not in this subset
    const std::span<T> &_original_set;
    vector<bool> _subset_repr;
    int length;

    class Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using reference = T &;

        pointer m_ptr; // current element
        Subset<ContainerType> &_subset;
        int curr_index;

      public:
        Iterator(pointer ptr, Subset<ContainerType> &_subset, int current_index)
            : m_ptr(ptr), _subset(_subset), curr_index(current_index) {}
        reference operator*() { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        // Prefix increment
        Iterator &operator++() {
            auto it = std::find(_subset._subset_repr.begin() + curr_index + 1,
                                _subset._subset_repr.end(), true);

            if (it == _subset._subset_repr.end()) {
                this->m_ptr = nullptr;
                this->curr_index = -1;
            } else {
                // WARN: This function should be noexcept, but I still can't
                // make sure this ->at() won't panic
                this->curr_index =
                    std::distance(_subset._subset_repr.begin(), it);
                this->m_ptr = &(_subset._original_set[this->curr_index]);
            }
            return *this;
        }

        // Postfix increment -
        // https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp
        Iterator operator++(int) {
            Iterator copy(m_ptr, _subset, curr_index);
            ++copy;
            return copy;
        }

        bool operator==(const Iterator &it) { return this->m_ptr == it.m_ptr; }
    };

  public:
    Iterator begin() {
        auto it = std::ranges::find(_subset_repr, true);
        if (it == _subset_repr.end()) {
            return Iterator(nullptr, *this, -1);
        } else {
            auto idx = std::distance(_subset_repr.begin(), it);
            return Iterator(&(_original_set[idx]), *this, idx);
        }
    }
    Iterator end() { return Iterator(nullptr, *this, -1); }

    void insert_index(size_t element_index) noexcept {
        if (element_index < _original_set.size()) {
            if (_subset_repr.size() < element_index) {
                // Making sure _subset_repr has at least 'element_index' values
                _subset_repr.resize(element_index);
            }
            _subset_repr[element_index] = true;
        }
    }

    int size() const { return length; }
    bool empty() const { return size() == 0; }

    Subset(const ContainerType &original_set) : _original_set(original_set) {

        // Subset array can be lesser size than that of original set (since it
        // just needs to be as long to store the 'last' true)
        length = 0;
    }

    friend class Iterator;
};
