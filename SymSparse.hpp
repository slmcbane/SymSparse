/*
 * Copyright 2020 The University of Texas at Austin.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef SYMSPARSE_HPP
#define SYMSPARSE_HPP

#include "SmallVector.hpp"

#include <algorithm>
#include <cstddef> // std::size_t;
#include <tuple>
#include <vector>

namespace SymSparse
{

template <class T, std::size_t MaxPerRow>
class SymmetricSparseMatrix
{
public:
    typedef smv::SmallVector<std::tuple<std::size_t, T>, MaxPerRow+1> small_vec;

    SymmetricSparseMatrix(std::size_t nrows) :
        m_rows(static_cast<decltype(m_rows)::size_type>(nrows))
    {}
    
    template <class Container>
    SymmetricSparseMatrix(std::size_t nrows, const Container &entries) :
        m_rows(static_cast<decltype(m_rows)::size_type>(nrows))
    {
        // Add all entries to appropriate rows.
        // Swap row and column to store everything as upper triangular.
        for (const auto &entry: entries)
        {
            std::size_t row = get<0>(entry);
            std::size_t col = get<1>(entry);
            if (col < row)
            {
                std::swap(row, col);
            }
            T val = get<2>(entry);

            rows[row].push_back(std::make_tuple(col, val));
        }

        // Sort each row, then combine duplicates
        for (auto &row: m_rows)
        {
            std::sort(row.begin(), row.end(),
                    [](auto e1, auto e2) { return get<0>(e1) < get<0>(e2); });
            combine_duplicates(row);
        }
    }

private:
    std::vector<small_vec> m_rows;

    void combine_duplicates(small_vec &row) noexcept
    {
        std::size_t i = 0, j;
        while (i < row.size() - 1)
        {
            j = i+1;
            while (j < row.size() && get<0>(row[j]) == get<0>(row[i]))
            {
                get<1>(row[i]) += get<1>(row[j]);
                j += 1;
            }
            i = j;
        }
        auto new_end = std::unique(row.begin(), row.end(),
                [](auto e1, auto e2) { return get<0>(e1) == get<0>(e2); });
        row.erase(new_end, row.end());
    }
};

} // namespace SymSparse

#endif /* SYMSPARSE_HPP */

