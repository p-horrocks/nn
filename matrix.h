#ifndef MATRIX_H
#define MATRIX_H

#include "funcs.h"

class Matrix
{
public:
    Matrix() {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    fpt value(int r, int c) const { return m_[c + (r * cols_)]; }

    void resize(int rows, int cols, bool randomise);
    fpt_vect multiply(const fpt_vect& a) const;
    void add(const Matrix& a);

private:
    int rows_ = 0;
    int cols_ = 0;
    fpt_vect m_;
};

std::ostream& operator << (std::ostream& os, const Matrix& m);

#endif // MATRIX_H
