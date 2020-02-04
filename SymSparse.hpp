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
#include <exception>
#include <tuple>
#include <type_traits>
#include <vector>

namespace SymSparse
{

struct OutOfBoundsIndex : public std::exception
{
    const std::size_t nrows, index;
};

/*
 * A square, symmetric sparse matrix.
 *
 * This class stores the upper triangular part of a sparse symmetric matrix.
 * The format is a list of rows, where each row is stored as a list of entries
 * in that row (as tuples (col, val)).
 *
 * Template parameters
 * -------------------
 *  - T is the type of the stored entries
 *  - MaxPerRow is the maximum number of __off diagonal__ entries in each row
 *
 * Invariants
 * ----------
 *  - Entries in each row are stored in sorted order, by column.
 *  - There is at most 1 entry in each column within a row.
 *  - No entries are stored with column < row.
 */
template <class T, std::size_t MaxPerRow>
class SymmetricSparseMatrix
{
public:
    typedef smv::SmallVector<std::tuple<std::size_t, T>, MaxPerRow+1> small_vec;
    
    // Initialize matrix with appropriate number of rows, but no entries
    SymmetricSparseMatrix(std::size_t nrows) :
        m_rows(static_cast<decltype(m_rows)::size_type>(nrows))
    {}
   
    /*
     * Initialize matrix from a list of entries
     *
     * Arguments:
     *   - nrows is the number of rows (and columns) in the matrix
     *   - Container is an iterable container. The elements in `entries` are
     *     accessible using a function template `get<i>` discoverable via ADL;
     *     I imagine this being `std::tuple`. The entries are in the format
     *     `(row, col, val)`. Entries in the lower triangular part are instead
     *     put into upper triangular format, and multiple entries in the same
     *     row and column have their values summed.
     *   - check_indices is a std::bool_constant; if it is `std::true_type{}`
     *     (default) then the values of `row` and `col` are checked to make sure
     *     they are in bounds given the size `nrows`.
     *
     * The constructed matrix has rows in sorted order, and each row and column
     * has either 0 or 1 entry.
     * 
     * If `check_indices` is enabled (default) and an index is out of bounds, an
     * `OutOfBoundsIndex` exception is thrown.
     */
    template <class Container, class CheckBounds = std::true_type>
    SymmetricSparseMatrix(std::size_t nrows, const Container &entries,
                          CheckBounds check_indices = CheckBounds{}) :
        m_rows(static_cast<decltype(m_rows)::size_type>(nrows))
    {
        // Add all entries to appropriate rows.
        // Swap row and column to store everything as upper triangular.
        for (const auto &entry: entries)
        {
            std::size_t row = get<0>(entry);
            std::size_t col = get<1>(entry);

            // Bounds check?
            if (CheckBounds::value)
            {
                if (row >= nrows)
                {
                    throw OutOfBoundsIndex{row};
                }
                if (col >= nrows)
                {
                    throw OutOfBoundsIndex{col};
                }
            }
            
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

/********************************************************************************
 * This block of code is tests, compiled if doctest.hpp is included
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("Check constructor of SymmetricSparseMatrix")
{

} // TEST_CASE

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End tests
 *******************************************************************************/

} // namespace SymSparse

#endif /* SYMSPARSE_HPP */

