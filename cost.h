#ifndef COST_H
#define COST_H

#include <memory>

#include "matrix.h"

class AbstractCost
{
public:
    AbstractCost() {}
    virtual ~AbstractCost() {}

    // Calculate the cost for activation a and output y
    virtual Matrix cost(const Matrix& a, const Matrix& y) = 0;

    // Return the error delta from the output y compared to activation a and
    // z vectors
    virtual Matrix outputError(const Matrix& z, const Matrix& a, const fpt_vect& y) = 0;
};
typedef std::shared_ptr<AbstractCost> AbstractCostPtr;

class QuadraticCost :
        public AbstractCost
{
public:
    Matrix cost(const Matrix& a, const Matrix& y) override;
    Matrix outputError(const Matrix& z, const Matrix& a, const fpt_vect& y) override;
};

class CrossEntropyCost :
        public AbstractCost
{
public:
    Matrix cost(const Matrix& a, const Matrix& y) override;
    Matrix outputError(const Matrix& z, const Matrix& a, const fpt_vect& y) override;
};

#endif // COST_H
