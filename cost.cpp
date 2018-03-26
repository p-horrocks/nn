#include "cost.h"

#include "matrix.h"

Matrix QuadraticCost::cost(const Matrix& a, const Matrix& y)
{
    return Matrix();
}

Matrix QuadraticCost::outputError(const Matrix& z, const Matrix& a, const fpt_vect& y)
{
    Matrix retval;
    retval.resize(a.rows(), 1, false);

    retval.apply([&](int r, int c, fpt){
        const fpt a1 = a.value(r, c);
        const fpt y1 = y[r];
        const fpt z1 = z.value(r, c);
        return (a1 - y1) * sigmoidPrime(z1);
    });

    return retval;
}

Matrix CrossEntropyCost::cost(const Matrix& a, const Matrix& y)
{
    return Matrix();
}

Matrix CrossEntropyCost::outputError(const Matrix&, const Matrix& a, const fpt_vect& y)
{
    Matrix retval;
    retval.resize(a.rows(), 1, false);

    retval.apply([&](int r, int c, fpt){
        const fpt a1 = a.value(r, c);
        const fpt y1 = y[r];
        return (a1 - y1);
    });

    return retval;
}
