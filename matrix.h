#ifndef MATRIX_H
#define MATRIX_H

#include <functional>

#include "funcs.h"

class Matrix
{
public:
    Matrix() {}
    Matrix(const Matrix& o);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    std::string shape() const;

    fpt value(int r, int c) const;
    void set(int r, int c, fpt v);

    void appendRow(const fpt_vect& row);

    void resize(int rows, int cols, bool randomise);
    void resize(int rows, int cols, const std::initializer_list<fpt>& init);

    // Create a single column matrix from the vector
    void fromVector(const fpt_vect& v);
    fpt_vect toVector() const;

    // Return the transpose of this matrix
    Matrix transpose() const;

    // Matrix multiplication. Returns this.a
    Matrix multiply(const Matrix& a) const;

    // Element-wise addition
    void add(const Matrix& m);

    // Apply a function per-element. The return value of the function is
    // assigned to each element. Function takes in row, column, value
    void apply(const std::function<fpt (int, int, fpt)>& f);

    // This is only used for testing. The epsilon must be large enough to
    // allow for the precision lost by the serialisation of the data
    bool isEqual(const Matrix& o, fpt eps = 1e-6) const;

private:
    int rows_ = 0;
    int cols_ = 0;
    fpt_vect m_;
};

std::ostream& operator << (std::ostream& os, const Matrix& m);

#endif // MATRIX_H
