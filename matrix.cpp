#include "matrix.h"

Matrix::Matrix(const Matrix& o)
{
    rows_ = o.rows_;
    cols_ = o.cols_;
    m_.resize(rows_ * cols_);
    std::copy(o.m_.begin(), o.m_.end(), m_.begin());
}

void Matrix::set(int r, int c, fpt v)
{
    m_[c + (r * cols_)] = v;
}

void Matrix::appendRow(const fpt_vect& row)
{
    assert(row.size() == cols_);
    ++rows_;
    m_.insert(m_.end(), row.begin(), row.end());
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

Matrix Matrix::transpose() const
{
    Matrix retval;
    retval.resize(cols_, rows_,false);

    for(int j = 0; j < cols_; ++j)
    {
        for(int i = 0; i < rows_; ++i)
        {
            fpt v = value(i, j);
            retval.set(j, i, v);
        }
    }

    return retval;
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

void Matrix::add(const Matrix& a)
{
    assert(rows_ == a.rows_);
    assert(cols_ == a.cols_);
    ::add(m_, a.m_);
}

void Matrix::apply(const std::function<fpt (int, int, fpt)>& f)
{
    int n = 0;
    for(int j = 0; j < cols_; ++j)
    {
        for(int i = 0; i < rows_; ++i, ++n)
        {
            m_[n] = f(i, j, m_[n]);
        }
    }
}

bool Matrix::isEqual(const Matrix* o, fpt eps) const
{
    if(o->rows_ != rows_)
        return false;

    if(o->cols_ != cols_)
        return false;

    for(int i = 0; i < m_.size(); ++i)
    {
        if(std::fabs(m_[i] - o->m_[i]) > eps)
            return false;
    }
    return true;
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
