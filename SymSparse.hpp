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
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
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

    OutOfBoundsIndex(std::size_t n, std::size_t i) noexcept : nrows{n}, index{i} {}

    const char *what() const noexcept
    {
        return "Out of bounds index in matrix operation; members nrows and index have size and "
               "index";
    }
};

struct DimensionMismatch : public std::exception
{
    std::size_t expected, actual;

    DimensionMismatch(std::size_t e, std::size_t a) noexcept : expected{e}, actual{a} {}

    const char *what() const noexcept
    {
        return "Dimension mismatch in matrix operation; members expected and actual store sizes";
    }
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
    typedef smv::SmallVector<std::tuple<std::size_t, T>, MaxPerRow + 1, SmallVecSizeCheck>
        small_vec;

    // Initialize matrix with appropriate number of rows, but no entries
    SymmetricSparseMatrix(std::size_t nrows)
        : m_rows(static_cast<typename decltype(m_rows)::size_type>(nrows))
    {
    }

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
    SymmetricSparseMatrix(
        std::size_t nrows, const Container &entries, CheckBounds check_indices = CheckBounds{})
        : m_rows(static_cast<typename decltype(m_rows)::size_type>(nrows))
    {
        // Add all entries to appropriate rows.
        // Swap row and column to store everything as upper triangular.
        for (const auto &entry : entries)
        {
            std::size_t row = get<0>(entry);
            std::size_t col = get<1>(entry);

            // Bounds check?
            if (check_indices)
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
        for (auto &row : m_rows)
        {
            using Entry = std::tuple<std::size_t, T>;
            std::sort(
                row.begin(), row.end(),
                [](Entry e1, Entry e2) { return get<0>(e1) < get<0>(e2); });
            combine_duplicates(row);
        }
    } // Container constructor

    template <class CheckBounds = std::true_type>
    const small_vec &row(std::size_t i, CheckBounds check_bounds = CheckBounds{}) const
    {
        if (check_bounds)
        {
            if (i >= m_rows.size())
            {
                throw OutOfBoundsIndex{m_rows.size(), i};
            }
        }

        return m_rows[i];
    }

    template <class CheckBounds = std::true_type>
    void
    insert_entry(std::size_t i, std::size_t j, T val, CheckBounds check_bounds = CheckBounds{})
    {
        if (check_bounds)
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

    template <class RHS, class Adjacent, class CheckBounds = std::true_type>
    void eliminate_dof(
        std::size_t index, T val, T scale, RHS &rhs, const Adjacent &adjacent,
        CheckBounds check_bounds = CheckBounds{})
    {
        if (check_bounds && (index >= m_rows.size()))
        {
            throw OutOfBoundsIndex{m_rows.size(), index};
        }

        for (auto row : adjacent)
        {
            if (check_bounds && (static_cast<std::size_t>(row) >= m_rows.size()))
            {
                throw OutOfBoundsIndex{m_rows.size(), index};
            }

            if (static_cast<std::size_t>(row) >= index)
            {
                continue;
            }

            auto it = m_rows[row].begin();
            while (it != m_rows[row].end() && get<0>(*it) < index)
            {
                it += 1;
            }

            if (it != m_rows[row].end() && get<0>(*it) == index)
            {
                rhs.row(row).array() -= get<1>(*it) * val;
                m_rows[row].erase(it);
            }
        }

        auto it = m_rows[index].begin();
        while (it != m_rows[index].end())
        {
            auto row = get<0>(*it);
            if (row == index)
            {
                it += 1;
            }
            else
            {
                rhs.row(row).array() -= get<1>(*it) * val;
                it = m_rows[index].erase(it);
            }
        }

        m_rows[index].clear();
        m_rows[index].push_back(std::make_tuple(index, scale));
        rhs.row(index).array() = val * scale;
    }

    template <class Adjacent, class CheckBounds = std::true_type>
    void eliminate_dof(
        std::size_t index, T val, T scale, const Adjacent &adjacent,
        CheckBounds check_bounds = CheckBounds{})
    {
        if (check_bounds && (index >= m_rows.size()))
        {
            throw OutOfBoundsIndex{m_rows.size(), index};
        }

        for (auto row : adjacent)
        {
            if (check_bounds && (static_cast<std::size_t>(row) >= m_rows.size()))
            {
                throw OutOfBoundsIndex{m_rows.size(), index};
            }

            if (static_cast<std::size_t>(row) >= index)
            {
                continue;
            }

            auto it = m_rows[row].begin();
            while (it != m_rows[row].end() && get<0>(*it) < index)
            {
                it += 1;
            }

            if (it != m_rows[row].end() && get<0>(*it) == index)
            {
                m_rows[row].erase(it);
            }
        }

        auto it = m_rows[index].begin();
        while (it != m_rows[index].end())
        {
            auto row = get<0>(*it);
            if (row == index)
            {
                it += 1;
            }
            else
            {
                it = m_rows[index].erase(it);
            }
        }

        m_rows[index].clear();
        m_rows[index].push_back(std::make_tuple(index, scale));
    }

    template <class X, class Y, class CheckSizes = std::true_type>
    Y &mul(const X &x, Y &y, CheckSizes size_check = CheckSizes{}) const
    {
        if (size_check)
        {
            if (x.size() != m_rows.size())
            {
                throw DimensionMismatch(m_rows.size(), x.size());
            }
            if (y.size() != m_rows.size())
            {
                throw DimensionMismatch(m_rows.size(), y.size());
            }
        }

        // Zero y (Hopefully this optimizes to a memset call; I'd like to use
        // std::fill but Eigen provides at least one example of vector objects
        // that don't implement the iterator interface for the STL).
        for (std::size_t i = 0; i < m_rows.size(); ++i)
        {
            y[i] = static_cast<T>(0);
        }

        for (std::size_t i = 0; i < m_rows.size(); ++i)
        {
            for (const auto &entry : m_rows[i])
            {
                const auto j = get<0>(entry);
                const auto a_ij = get<1>(entry);
                y[i] += a_ij * x[j];
                if (j != i)
                {
                    y[j] += a_ij * x[i];
                }
            }
        }
        return y;
    }

    template <class X, class Y, class CheckSizes = std::true_type>
    Y &mul(T a, const X &x, T b, Y &y, CheckSizes size_check = CheckSizes{}) const
    {
        if (size_check)
        {
            if (x.size() != m_rows.size())
            {
                throw DimensionMismatch(m_rows.size(), x.size());
            }
            if (y.size() != m_rows.size())
            {
                throw DimensionMismatch(m_rows.size(), y.size());
            }
        }

        for (std::size_t i = 0; i < m_rows.size(); ++i)
        {
            y[i] = b * y[i];
        }

        for (std::size_t i = 0; i < m_rows.size(); ++i)
        {
            for (const auto &entry : m_rows[i])
            {
                const auto j = get<0>(entry);
                const auto a_times_aij = a * get<1>(entry);
                y[i] += a_times_aij * x[j];
                if (j != i)
                {
                    y[j] += a_times_aij * x[i];
                }
            }
        }
        return y;
    }

    size_t num_rows() const noexcept { return m_rows.size(); }

  private:
    std::vector<small_vec> m_rows;

    static void combine_duplicates(small_vec &row) noexcept
    {
        using Entry = std::tuple<std::size_t, T>;
        if (row.size() == 0)
        {
            return;
        }

        std::size_t i = 0, j;
        while (i < row.size() - 1)
        {
            j = i + 1;
            while (j < row.size() && get<0>(row[j]) == get<0>(row[i]))
            {
                get<1>(row[i]) += get<1>(row[j]);
                j += 1;
            }
            i = j;
        }
        auto new_end = std::unique(
            row.begin(), row.end(), [](Entry e1, Entry e2) { return get<0>(e1) == get<0>(e2); });
        row.erase(new_end, row.end());
    }

    void insert_entry_in_row(std::size_t i, std::size_t j, T val)
    {
        auto &row = m_rows[i];
        auto dst_iterator = std::find_if(
            row.begin(), row.end(),
            [=](const std::tuple<size_t, T> &entry) { return get<0>(entry) >= j; });
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

// Trait specifying the magic number for a type to go in the serialization
// header of a serialized sparse matrix holding values of that type.
// All values should be >= 0. The value of -1 in the base type will disable
// compilation, as will any other values < 0.
template <class T>
struct type_number : std::integral_constant<int32_t, -1>
{
};

// Specializations.
template <>
struct type_number<uint8_t> : std::integral_constant<int32_t, 0>
{
};

template <>
struct type_number<float> : std::integral_constant<int32_t, 1>
{
};

template <>
struct type_number<double> : std::integral_constant<int32_t, 2>
{
};

template <>
struct type_number<int> : std::integral_constant<int32_t, 3>
{
};

template <class T>
struct has_valid_type_number
    : std::integral_constant<
          bool, std::is_same<int32_t, typename type_number<T>::value_type>::value &&
                    type_number<T>::value >= 0>
{
};

struct SerializationException : public std::exception
{
    const char *msg;
    const char *what() const noexcept { return msg; }

    SerializationException(const char *m) noexcept : msg{m} {}
};

/*
 * Serialize a matrix A to file "dst".
 * For types not included in the specializations above, YOU will need to define
 * one for serialization and deserialization to be enabled.
 *
 * Serialization format
 * ====================
 * - A 32-bit signed integer specifying the type of entries in the matrix
 * - A 64-bit unsigned integer specifying the number of rows in the matrix
 * - For each row, a 64-bit unsigned integer that gives the number of entries
 *   in that row.
 * - The columns of each entry as 64-bit unsigned integers; these are in one
 *   block in order to optimize IO performance.
 * - The values of each entry in this platform's canonical binary format.
 */
template <class T, size_t N, bool B>
typename std::enable_if<has_valid_type_number<T>::value>::type
serialize(FILE *dst, const SymmetricSparseMatrix<T, N, B> &A)
{
    constexpr int32_t typecode = type_number<T>::value;
    fwrite(&typecode, sizeof(int32_t), 1, dst);
    uint64_t nrows = A.num_rows();
    fwrite(&nrows, sizeof(uint64_t), 1, dst);

    std::vector<uint64_t> ibuf;
    std::vector<T> dbuf;
    ibuf.reserve(nrows);

    size_t nentries = 0;
    for (size_t i = 0; i < nrows; ++i)
    {
        ibuf.push_back(A.row(i).size());
        nentries += A.row(i).size();
    }

    fwrite(ibuf.data(), sizeof(uint64_t), ibuf.size(), dst);
    ibuf.clear();
    ibuf.reserve(nentries);
    dbuf.reserve(nentries);

    for (size_t i = 0; i < nrows; ++i)
    {
        const auto &row = A.row(i);
        for (const auto &entry : row)
        {
            ibuf.push_back(std::get<0>(entry));
            dbuf.push_back(std::get<1>(entry));
        }
    }

    fwrite(ibuf.data(), sizeof(uint64_t), ibuf.size(), dst);
    fwrite(dbuf.data(), sizeof(T), dbuf.size(), dst);
}

template <class T, size_t N>
typename std::enable_if<has_valid_type_number<T>::value, SymmetricSparseMatrix<T, N>>::type
deserialize(FILE *src)
{
    int32_t typecode;
    std::vector<std::tuple<std::size_t, std::size_t, T>> entries;

    fread(&typecode, sizeof(int32_t), 1, src);
    if (typecode != type_number<T>::value)
    {
        throw SerializationException("Got unknown type code");
    }

    uint64_t nrows;
    fread(&nrows, sizeof(uint64_t), 1, src);
    std::vector<uint64_t> ibuf(static_cast<typename std::vector<uint64_t>::size_type>(nrows));
    fread(ibuf.data(), sizeof(uint64_t), ibuf.size(), src);

    auto nentries = std::accumulate(ibuf.begin(), ibuf.end(), static_cast<uint64_t>(0));
    entries.resize(nentries);
    size_t index = 0;
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < ibuf[i]; ++j)
        {
            std::get<0>(entries[index++]) = i;
        }
    }

    ibuf.resize(nentries);
    fread(ibuf.data(), sizeof(uint64_t), ibuf.size(), src);

    for (size_t i = 0; i < nentries; ++i)
    {
        std::get<1>(entries[i]) = ibuf[i];
    }
    ibuf = std::vector<uint64_t>();

    std::vector<T> dbuf(static_cast<typename std::vector<T>::size_type>(nentries));
    fread(dbuf.data(), sizeof(T), dbuf.size(), src);
    for (size_t i = 0; i < nentries; ++i)
    {
        std::get<2>(entries[i]) = dbuf[i];
    }
    dbuf = std::vector<T>();

    return SymmetricSparseMatrix<T, N>(nrows, entries);
}

template <class T, size_t N1, size_t N2, bool B1, bool B2>
bool operator==(
    const SymmetricSparseMatrix<T, N1, B1> &A, const SymmetricSparseMatrix<T, N2, B2> &B) noexcept
{
    if (A.num_rows() != B.num_rows())
    {
        return false;
    }

    for (size_t i = 0; i < A.num_rows(); ++i)
    {
        const auto &Arow = A.row(i);
        const auto &Brow = B.row(i);

        for (auto pair = std::make_pair(Arow.begin(), Brow.begin());
             pair.first != Arow.end() && pair.second != Brow.end(); ++pair.first, ++pair.second)
        {
            if (*pair.first != *pair.second)
            {
                return false;
            }
        }
    }
    return true;
}

/********************************************************************************
 * This block of code is tests, compiled if doctest.hpp is included
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("Test constructor of SymmetricSparseMatrix")
{
    SUBCASE("No funny business, just a list of entries in sorted order")
    {
        std::array<std::array<int, 3>, 10> entries = {0, 0, 1, 0, 1, 2, 0, 2, 3, 0,
                                                      3, 4, 1, 1, 2, 1, 2, 3, 1, 3,
                                                      4, 2, 2, 3, 2, 3, 4, 3, 3, 4};

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
            std::make_tuple(0, 0, 1), std::make_tuple(0, 1, 1), std::make_tuple(0, 2, 3),
            std::make_tuple(0, 3, 4), std::make_tuple(0, 1, 1), std::make_tuple(1, 1, 1),
            std::make_tuple(1, 2, 3), std::make_tuple(1, 3, 4), std::make_tuple(1, 1, 1),
            std::make_tuple(2, 2, 3), std::make_tuple(2, 3, 4), std::make_tuple(3, 3, 4)};

        auto construct_bad = [&]() { return SymmetricSparseMatrix<int, 3>(4, entries); };
        auto construct_bad_but_nothrow = [&]()
        { return SymmetricSparseMatrix<int, 3, false>(4, entries); };
        auto construct_good = [&]() { return SymmetricSparseMatrix<int, 4>(4, entries); };

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
            std::make_tuple(0, 0, 1), std::make_tuple(0, 1, 1), std::make_tuple(1, 0, 1),
            std::make_tuple(0, 2, 1), std::make_tuple(2, 0, 1), std::make_tuple(0, 2, 1),
            std::make_tuple(0, 3, 2), std::make_tuple(3, 0, 2), std::make_tuple(1, 1, 1),
            std::make_tuple(1, 1, 1), std::make_tuple(1, 2, 2), std::make_tuple(2, 1, 1),
            std::make_tuple(1, 3, 4), std::make_tuple(2, 2, 3), std::make_tuple(3, 2, 4),
            std::make_tuple(3, 3, 4)};

        std::shuffle(entries.begin(), entries.end(), std::default_random_engine());

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
    std::array<std::array<int, 3>, 1> entry = {0, 3, 4};

    auto construct_bad = [&]() { return SymmetricSparseMatrix<int, 1>(3, entry); };

    REQUIRE_THROWS_AS(construct_bad(), OutOfBoundsIndex);

    auto construct_ok = [&]()
    { return SymmetricSparseMatrix<int, 1>(3, entry, std::false_type{}); };

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
            std::make_tuple(0, 0, 1), std::make_tuple(0, 1, 1), std::make_tuple(1, 0, 1),
            std::make_tuple(0, 2, 1), std::make_tuple(2, 0, 1), std::make_tuple(0, 2, 1),
            std::make_tuple(0, 3, 2), std::make_tuple(3, 0, 2), std::make_tuple(1, 1, 1),
            std::make_tuple(1, 1, 1), std::make_tuple(1, 2, 2), std::make_tuple(2, 1, 1),
            std::make_tuple(1, 3, 4), std::make_tuple(2, 2, 3), std::make_tuple(3, 2, 4),
            std::make_tuple(3, 3, 4)};

        std::shuffle(entries.begin(), entries.end(), std::default_random_engine());

        SymmetricSparseMatrix<int, 3> A(4);
        for (const auto &entry : entries)
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

TEST_CASE("Test eliminating degrees of freedom")
{
    std::array<std::array<int, 3>, 16> entries = {0, 0, 2, 0, 3, 5, 0, 5, 1, 0, 6, 3, 1, 1, 3, 1,
                                                  2, 2, 1, 4, 3, 1, 6, 1, 2, 2, 4, 2, 3, 1, 3, 3,
                                                  1, 3, 5, 6, 3, 6, 5, 4, 4, 3, 5, 5, 2, 6, 6, 5};

    std::array<int, 7> rhs = {1, 2, 3, 4, 5, 6, 7};

    SymmetricSparseMatrix<int, 3> A(7, entries);

    SUBCASE("Eliminate degree of freedom given only the shared neighbors")
    {
        A.eliminate_dof(3, 3, 1, rhs, std::array<int, 4>{0, 2, 5, 6});
        REQUIRE(rhs == std::array<int, 7>{-14, 2, 0, 3, 5, -12, -8});

        {
            const auto &row = A.row(0);

            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 5);
            REQUIRE(get<1>(row[1]) == 1);
            REQUIRE(get<0>(row[2]) == 6);
            REQUIRE(get<1>(row[2]) == 3);
        }

        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 4);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 6);
            REQUIRE(get<1>(row[3]) == 1);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 1);
        }

        {
            const auto &row = A.row(4);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 4);
            REQUIRE(get<1>(row[0]) == 3);
        }

        {
            const auto &row = A.row(5);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 5);
            REQUIRE(get<1>(row[0]) == 2);
        }

        {
            const auto &row = A.row(6);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 6);
            REQUIRE(get<1>(row[0]) == 5);
        }
    } // SUBCASE

    SUBCASE("Test eliminating dof given extraneous neighbors")
    {
        A.eliminate_dof(3, 3, 1, rhs, std::array<int, 7>{0, 1, 2, 3, 4, 5, 6});
        REQUIRE(rhs == std::array<int, 7>{-14, 2, 0, 3, 5, -12, -8});

        {
            const auto &row = A.row(0);

            REQUIRE(row.size() == 3);
            REQUIRE(get<0>(row[0]) == 0);
            REQUIRE(get<1>(row[0]) == 2);
            REQUIRE(get<0>(row[1]) == 5);
            REQUIRE(get<1>(row[1]) == 1);
            REQUIRE(get<0>(row[2]) == 6);
            REQUIRE(get<1>(row[2]) == 3);
        }

        {
            const auto &row = A.row(1);
            REQUIRE(row.size() == 4);
            REQUIRE(get<0>(row[0]) == 1);
            REQUIRE(get<1>(row[0]) == 3);
            REQUIRE(get<0>(row[1]) == 2);
            REQUIRE(get<1>(row[1]) == 2);
            REQUIRE(get<0>(row[2]) == 4);
            REQUIRE(get<1>(row[2]) == 3);
            REQUIRE(get<0>(row[3]) == 6);
            REQUIRE(get<1>(row[3]) == 1);
        }

        {
            const auto &row = A.row(2);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 2);
            REQUIRE(get<1>(row[0]) == 4);
        }

        {
            const auto &row = A.row(3);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 3);
            REQUIRE(get<1>(row[0]) == 1);
        }

        {
            const auto &row = A.row(4);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 4);
            REQUIRE(get<1>(row[0]) == 3);
        }

        {
            const auto &row = A.row(5);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 5);
            REQUIRE(get<1>(row[0]) == 2);
        }

        {
            const auto &row = A.row(6);
            REQUIRE(row.size() == 1);
            REQUIRE(get<0>(row[0]) == 6);
            REQUIRE(get<1>(row[0]) == 5);
        }
    }
} // TEST_CASE

TEST_CASE("Test multiplication by a vector")
{
    const std::array<std::array<int, 3>, 12> entries = {0, 0, 2, 0, 5, 1, 0, 6, 3, 1, 1, 3,
                                                        1, 2, 2, 1, 4, 3, 1, 6, 1, 2, 2, 4,
                                                        3, 3, 1, 4, 4, 3, 5, 5, 2, 6, 6, 5};

    const SymmetricSparseMatrix<int, 3> A(7, entries);

    const std::array<int, 7> rhs = {-14, 2, 0, 3, 5, -12, -8};
    std::array<int, 7> dst;

    // Test simple overwrite of dst
    A.mul(rhs, dst);

    REQUIRE(dst == std::array<int, 7>{-64, 13, 4, 3, 21, -38, -80});

    // Test with alpha and beta arguments (a * A * x + b * y)
    A.mul(3, rhs, -2, dst);

    REQUIRE(dst == std::array<int, 7>{-64, 13, 4, 3, 21, -38, -80});
} // TEST_CASE

TEST_CASE("Test serialization of SymSparse")
{
    std::array<std::array<int, 3>, 10> entries = {0, 0, 1, 0, 1, 2, 0, 2, 3, 0, 3, 4, 1, 1, 2,
                                                  1, 2, 3, 1, 3, 4, 2, 2, 3, 2, 3, 4, 3, 3, 4};

    SymmetricSparseMatrix<int, 3> A(4, entries);
    FILE *destination_file = fopen("serialized.dat", "w");
    serialize(destination_file, A);
    fclose(destination_file);
    FILE *src_file = fopen("serialized.dat", "r");
    auto B = deserialize<int, 3>(src_file);
    fclose(src_file);

    REQUIRE(A == B);
}

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End tests
 *******************************************************************************/

} // namespace SymSparse

#endif /* SYMSPARSE_HPP */

