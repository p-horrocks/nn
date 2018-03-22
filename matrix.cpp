#include "matrix.h"

void Matrix::set(int r, int c, fpt v)
{
    m_[c + (r * cols_)] = v;
}

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

void Matrix::resize(int rows, int cols, const std::initializer_list<fpt>& init)
{
    rows_ = rows;
    cols_ = cols;
    m_.resize(rows * cols);

    assert((init.end() - init.begin()) == m_.size());
    std::copy(init.begin(), init.end(), m_.begin());
}

void Matrix::fromVector(const fpt_vect& v)
{
    resize(v.size(), 1, false);
    std::copy(v.begin(), v.end(), m_.begin());
}

fpt_vect Matrix::toVector() const
{
    assert(cols_ == 1);
    return m_;
}

void Matrix::transpose()
{
    auto m = m_;
    std::swap(rows_, cols_);
    for(int i = 0; i < m_.size(); ++i)
    {
        int x = i % cols_;
        int y = i / cols_;
        int n = y + (x * rows_);
        m_[i] = m[n];
    }
}

Matrix Matrix::multiply(const Matrix& a) const
{
    assert(cols_ == a.rows_);

    Matrix retval;
    retval.resize(rows_, a.cols_, false);

    for(int j = 0; j < a.cols_; ++j)
    {
        for(int i = 0; i < rows_; ++i)
        {
            fpt v = 0;
            for(int n = 0; n < cols_; ++n)
            {
                v += value(i, n) * a.value(n, j);
            }
            retval.set(i, j, v);
        }
    }

    return retval;
}

/*remove-me
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
*/

void Matrix::add(const Matrix& a)
{
    assert(rows_ == a.rows_);
    assert(cols_ == a.cols_);
    ::add(m_, a.m_);
}

void Matrix::sigmoid()
{
    std::transform(m_.begin(), m_.end(), m_.begin(), &::sigmoid);
}

void Matrix::update(const Matrix& m, fpt factor)
{
    for(int i = 0; i < m_.size(); ++i)
    {
        m_[i] -= factor * m.m_[i];
    }
}

std::ostream& operator << (std::ostream& os, const Matrix& m)
{
    for(int j = 0; j < m.rows(); ++j)
    {
        os << "[ ";
        for(int i = 0; i < m.cols(); ++i)
        {
            os << m.value(j, i) << " ";
        }
        os << "]" << std::endl;
    }
    return os;
}
