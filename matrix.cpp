#include "matrix.h"

void Matrix::resize(int rows, int cols, bool randomise)
{
    rows_ = rows;
    cols_ = cols;
    m_.resize(rows * cols);
    if(randomise)
    {
        std::generate(m_.begin(), m_.end(), &normalRand);
    }
    else
    {
        std::fill(m_.begin(), m_.end(), (fpt)0);
    }
}

fpt_vect Matrix::multiply(const fpt_vect& a) const
{
    // Implements V = a.M
    // Where M is this matrix
    assert(a.size() == cols_);
    fpt_vect retval(rows_);
    for(int i = 0; i < rows_; ++i)
    {
        fpt v = 0.f;
        for(int j = 0; j < cols_; ++j)
        {
            v += a[j] * m_[j + (i * cols_)];
        }
        retval[i] = v;
    }
    return retval;
}

void Matrix::add(const Matrix& a)
{
    assert(rows_ == a.rows_);
    assert(cols_ == a.cols_);
    ::add(m_, a.m_);
}

std::ostream& operator << (std::ostream& os, const Matrix& m)
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
