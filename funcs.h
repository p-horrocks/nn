#ifndef FUNCS_H
#define FUNCS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

float normalRand();

template<typename T>
T sigmoid(T z)
{
    return (T)1 / ((T)1 + std::exp(-z));
}

template<typename T>
T sigmoidPrime(T z)
{
    return sigmoid(z) * ((T)1 - sigmoid(z));
}

template<typename T>
void sigmoid(std::vector<T>& a)
{
    std::transform(a.begin(), a.end(), a.begin(), [](T n){return sigmoid(n);});
}

template<typename T>
void add(std::vector<T>& r, const std::vector<T>& v)
{
    assert(r.size() == v.size());
    std::transform(r.begin(), r.end(), v.begin(), r.begin(), [](T a, T b){return (a + b);});
}

template<typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    for(int j = 0; j < v.size(); ++j)
    {
        os << "[ ";
        os << v[j];
        os << " ]" << std::endl;
    }

    return os;
}

#endif // FUNCS_H
