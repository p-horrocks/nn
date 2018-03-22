#include "funcs.h"

#include <random>
#include <sys/time.h>

std::default_random_engine* __gen = nullptr;
std::normal_distribution<fpt> __dist(0.f, 1.f);

fpt normalRand()
{
    if(!__gen)
    {
        struct timeval t;
        gettimeofday(&t, nullptr);
        __gen = new std::default_random_engine(t.tv_usec);
    }

    return __dist(*__gen);
}

fpt sigmoid(fpt z)
{
    return (fpt)1 / ((fpt)1 + std::exp(-z));
}

void sigmoid(fpt_vect& a)
{
    std::transform(a.begin(), a.end(), a.begin(), [](fpt n){return sigmoid(n);});
}

fpt sigmoidPrime(fpt z)
{
    return sigmoid(z) * ((fpt)1 - sigmoid(z));
}

void sigmoidPrime(fpt_vect& a)
{
    std::transform(a.begin(), a.end(), a.begin(), [](fpt n){return sigmoidPrime(n);});
}

void add(fpt_vect& r, const fpt_vect& v)
{
    assert(r.size() == v.size());
    std::transform(r.begin(), r.end(), v.begin(), r.begin(), [](fpt a, fpt b){return (a + b);});
}

void sub(fpt_vect& r, const fpt_vect& v)
{
    assert(r.size() == v.size());
    std::transform(r.begin(), r.end(), v.begin(), r.begin(), [](fpt a, fpt b){return (a - b);});
}

void dot(fpt_vect& r, const fpt_vect& v)
{
    assert(r.size() == v.size());
    std::transform(r.begin(), r.end(), v.begin(), r.begin(), [](fpt a, fpt b){return (a * b);});
}

void hardMax(fpt_vect& a)
{
    auto m = std::max_element(a.begin(), a.end());
    std::fill(a.begin(), a.end(), (fpt)0);
    *m = (fpt)1;
}

std::ostream& operator << (std::ostream& os, const fpt_vect& v)
{
    for(int j = 0; j < v.size(); ++j)
    {
        os << "[ ";
        os << v[j];
        os << " ]" << std::endl;
    }

    return os;
}
