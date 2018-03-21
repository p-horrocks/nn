#ifndef MATRIX_H
#define MATRIX_H

#include "funcs.h"

template<typename T>
class Matrix
{
public:
    Matrix() {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    T value(int r, int c) const { return m_[c + (r * cols_)]; }

    void resize(int rows, int cols)
    {
        rows_ = rows;
        cols_ = cols;
        m_.resize(rows * cols);
        std::generate(m_.begin(), m_.end(), &normalRand);
    }

    std::vector<T> multiply(const std::vector<T>& a) const
    {
        // Implements V = a.M
        // Where M is this matrix
        assert(a.size() == cols_);
        std::vector<T> retval(rows_);
        for(int i = 0; i < rows_; ++i)
        {
            float v = 0.f;
            for(int j = 0; j < cols_; ++j)
            {
                v += a[j] * m_[j + (i * cols_)];
            }
            retval[i] = v;
        }
        return retval;
    }

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<T> m_;
};

template<typename T>
std::ostream& operator << (std::ostream& os, const Matrix<T>& m)
{
    for(int j = 0; j < m.rows(); ++j)
    {
        os << "[ ";
        for(int i = 0; i < m.cols(); ++i)
        {
            os << m.value(i, j) << " ";
        }
        os << "]" << std::endl;
    }

    return os;
}

#endif // MATRIX_H
