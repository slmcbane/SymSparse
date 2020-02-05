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

using std::get;

#ifdef DOCTEST_LIBRARY_INCLUDED

#include <random>

#endif // DOCTEST_LIBRARY_INCLUDED

namespace SymSparse
{

struct OutOfBoundsIndex : public std::exception
{
    std::size_t nrows, index;

    OutOfBoundsIndex(std::size_t n, std::size_t i) noexcept : nrows{n}, index{i}
    {}
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
template <class T, std::size_t MaxPerRow, bool SmallVecSizeCheck = true>
class SymmetricSparseMatrix
{
public:
    typedef smv::SmallVector<std::tuple<std::size_t, T>,
                             MaxPerRow+1, SmallVecSizeCheck> small_vec;
    
    // Initialize matrix with appropriate number of rows, but no entries
    SymmetricSparseMatrix(std::size_t nrows) :
        m_rows(static_cast<typename decltype(m_rows)::size_type>(nrows))
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
        m_rows(static_cast<typename decltype(m_rows)::size_type>(nrows))
    {
        // Add all entries to appropriate rows.
        // Swap row and column to store everything as upper triangular.
        for (const auto &entry: entries)
        {
            std::size_t row = get<0>(entry);
            std::size_t col = get<1>(entry);

            // Bounds check?
            if (check_indices())
            {
                if (row >= nrows)
                {
                    throw OutOfBoundsIndex{nrows, row};
                }
                if (col >= nrows)
                {
                    throw OutOfBoundsIndex{nrows, col};
                }
            }
            
            if (col < row)
            {
                std::swap(row, col);
            }
            T val = get<2>(entry);

            m_rows[row].push_back(std::make_tuple(col, val));
        }

        // Sort each row, then combine duplicates
        for (auto &row: m_rows)
        {
            std::sort(row.begin(), row.end(),
                    [](auto e1, auto e2) { return get<0>(e1) < get<0>(e2); });
            combine_duplicates(row);
        }
    } // Container constructor

    template <class CheckBounds = std::true_type>
    const small_vec &row(std::size_t i,
            CheckBounds check_bounds = CheckBounds{}) const
    {
        if (check_bounds())
        {
            if (i >= m_rows.size())
            {
                throw OutOfBoundsIndex{m_rows.size(), i};
            }
        }

        return m_rows[i];
    }

    template <class CheckBounds = std::true_type>
    void insert_entry(std::size_t i, std::size_t j, T val,
            CheckBounds check_bounds = CheckBounds{})
    {
        if (check_bounds())
        {
            if (i >= m_rows.size())
            {
                throw OutOfBoundsIndex{m_rows.size(), i};
            }
            if (j >= m_rows.size())
            {
                throw OutOfBoundsIndex{m_rows.size(), j};
            }
        }

        if (j < i)
        {
            std::swap(i, j);
        }
        insert_entry_in_row(i, j, val);
    }

private:
    std::vector<small_vec> m_rows;

    static void combine_duplicates(small_vec &row) noexcept
    {
        if (row.size() == 0)
        {
            return;
        }

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

    void insert_entry_in_row(std::size_t i, std::size_t j, T val)
    {
        auto &row = m_rows[i];
        auto dst_iterator = std::find_if(row.begin(), row.end(),
                [=](const std::tuple<size_t, T> &entry)
                {
                    return get<0>(entry) >= j;
                }
        );
        if (dst_iterator == row.end())
        {
            row.push_back(std::make_tuple(j, val));
        }
        else if (get<0>(*dst_iterator) == j)
        {
            get<1>(*dst_iterator) += val;
        }
        else
        {
            row.insert(dst_iterator, std::make_tuple(j, val));
        }
    }
};

/********************************************************************************
 * This block of code is tests, compiled if doctest.hpp is included
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("Test constructor of SymmetricSparseMatrix")
{
    SUBCASE("No funny business, just a list of entries in sorted order")
    {
        std::array<std::array<int, 3>, 10> entries = {
            0, 0, 1, 0, 1, 2, 0, 2, 3, 0, 3, 4,
            1, 1, 2, 1, 2, 3, 1, 3, 4, 2, 2, 3,
            2, 3, 4, 3, 3, 4 };

        SymmetricSparseMatrix<int, 3> A(4, entries);

        {
            const auto &row = A.row(0);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 1);
            REQUIRE(get<0>(row[1]) == 1);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 2);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 3);
            REQUIRE(get<1>(row[3]) == 4);
        }
        
        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 3);
            REQUIRE(get<0>(row[2]) == 3);
            REQUIRE(get<1>(row[2]) == 4);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 2);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 3);
            REQUIRE(get<1>(row[1]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 4);
        }
    } // SUBCASE

    SUBCASE("All indices upper triangular, some need combined")
    {
        std::vector<std::tuple<int, int, int>> entries = {
            std::make_tuple(0, 0, 1),
            std::make_tuple(0, 1, 1),
            std::make_tuple(0, 2, 3),
            std::make_tuple(0, 3, 4),
            std::make_tuple(0, 1, 1),
            std::make_tuple(1, 1, 1),
            std::make_tuple(1, 2, 3),
            std::make_tuple(1, 3, 4),
            std::make_tuple(1, 1, 1),
            std::make_tuple(2, 2, 3),
            std::make_tuple(2, 3, 4),
            std::make_tuple(3, 3, 4)
        };

        auto construct_bad = [&]()
        {
            return SymmetricSparseMatrix<int, 3>(4, entries);
        };
        auto construct_bad_but_nothrow = [&]()
        {
            return SymmetricSparseMatrix<int, 3, false>(4, entries);
        };
        auto construct_good = [&]()
        {
            return SymmetricSparseMatrix<int, 4>(4, entries);
        };

        // By default this should throw because we will have extra elements in
        // the first row.
        REQUIRE_THROWS_AS(construct_bad(), smv::MaxSizeExceeded);
        REQUIRE_NOTHROW(construct_good());
        REQUIRE_NOTHROW(construct_bad_but_nothrow());

        // Check entries
        const SymmetricSparseMatrix<int, 4> A(4, entries);

        {
            const auto &row = A.row(0);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 1);
            REQUIRE(get<0>(row[1]) == 1);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 2);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 3);
            REQUIRE(get<1>(row[3]) == 4);
        }
        
        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 3);
            REQUIRE(get<0>(row[2]) == 3);
            REQUIRE(get<1>(row[2]) == 4);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 2);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 3);
            REQUIRE(get<1>(row[1]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 4);
        }

    } // SUBCASE

    SUBCASE("Test full combo, lower triangular indices + combining")
    {
        std::vector<std::tuple<int, int, int>> entries = {
            std::make_tuple(0, 0, 1),
            std::make_tuple(0, 1, 1),
            std::make_tuple(1, 0, 1),
            std::make_tuple(0, 2, 1),
            std::make_tuple(2, 0, 1),
            std::make_tuple(0, 2, 1),
            std::make_tuple(0, 3, 2),
            std::make_tuple(3, 0, 2),
            std::make_tuple(1, 1, 1),
            std::make_tuple(1, 1, 1),
            std::make_tuple(1, 2, 2),
            std::make_tuple(2, 1, 1),
            std::make_tuple(1, 3, 4),
            std::make_tuple(2, 2, 3),
            std::make_tuple(3, 2, 4),
            std::make_tuple(3, 3, 4)
        };

        std::shuffle(entries.begin(), entries.end(),
                     std::default_random_engine());

        const SymmetricSparseMatrix<int, 7> A(4, entries);

        {
            const auto &row = A.row(0);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 1);
            REQUIRE(get<0>(row[1]) == 1);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 2);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 3);
            REQUIRE(get<1>(row[3]) == 4);
        }
        
        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 3);
            REQUIRE(get<0>(row[2]) == 3);
            REQUIRE(get<1>(row[2]) == 4);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 2);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 3);
            REQUIRE(get<1>(row[1]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 4);
        }
    } // SUBCASE
} // TEST_CASE

TEST_CASE("Check that some exceptions get thrown as they should")
{
    std::array<std::array<int, 3>, 1> entry = { 0, 3, 4 };

    auto construct_bad = [&]()
    {
        return SymmetricSparseMatrix<int, 1>(3, entry);
    };

    REQUIRE_THROWS_AS(construct_bad(), OutOfBoundsIndex);

    auto construct_ok = [&]()
    {
        return SymmetricSparseMatrix<int, 1>(3, entry, std::false_type{});
    };

    REQUIRE_NOTHROW(construct_ok());
    {
        SymmetricSparseMatrix<int, 1> A(3, entry, std::false_type{});

        REQUIRE_NOTHROW(A.row(0));
        REQUIRE_NOTHROW(A.row(1));
        REQUIRE_NOTHROW(A.row(2));
        REQUIRE_NOTHROW(A.row(3, std::false_type{}));
        REQUIRE_THROWS_AS(A.row(3), OutOfBoundsIndex);
    }

    {
        SymmetricSparseMatrix<int, 1> A(3, entry, std::false_type{});

        REQUIRE_NOTHROW(A.insert_entry(1, 0, 1));
        REQUIRE_NOTHROW(A.insert_entry(1, 1, 1));
        REQUIRE_THROWS_AS(A.insert_entry(1, 3, 0), OutOfBoundsIndex);
        REQUIRE_NOTHROW(A.insert_entry(1, 3, 0, std::false_type{}));
        REQUIRE_THROWS_AS(A.insert_entry(1, 2, 0), smv::MaxSizeExceeded);
    }
} // TEST_CASE

TEST_CASE("Test constructing a matrix using successive calls to insert")
{
    SUBCASE("Test full combo, lower triangular indices + combining")
    {
        std::vector<std::tuple<int, int, int>> entries = {
            std::make_tuple(0, 0, 1),
            std::make_tuple(0, 1, 1),
            std::make_tuple(1, 0, 1),
            std::make_tuple(0, 2, 1),
            std::make_tuple(2, 0, 1),
            std::make_tuple(0, 2, 1),
            std::make_tuple(0, 3, 2),
            std::make_tuple(3, 0, 2),
            std::make_tuple(1, 1, 1),
            std::make_tuple(1, 1, 1),
            std::make_tuple(1, 2, 2),
            std::make_tuple(2, 1, 1),
            std::make_tuple(1, 3, 4),
            std::make_tuple(2, 2, 3),
            std::make_tuple(3, 2, 4),
            std::make_tuple(3, 3, 4)
        };

        std::shuffle(entries.begin(), entries.end(),
                     std::default_random_engine());

        SymmetricSparseMatrix<int, 3> A(4);
        for (const auto &entry: entries)
        {
            A.insert_entry(get<0>(entry), get<1>(entry), get<2>(entry));
        }

        {
            const auto &row = A.row(0);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 1);
            REQUIRE(get<0>(row[1]) == 1);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 2);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 3);
            REQUIRE(get<1>(row[3]) == 4);
        }
        
        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 3);
            REQUIRE(get<0>(row[2]) == 3);
            REQUIRE(get<1>(row[2]) == 4);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 2);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 3);
            REQUIRE(get<1>(row[1]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 4);
        }
    } // SUBCASE
} // TEST_CASE

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End tests
 *******************************************************************************/

} // namespace SymSparse

#endif /* SYMSPARSE_HPP */

