#include "funcs.h"

#include <random>
#include <sys/time.h>

std::default_random_engine* __gen = nullptr;
std::normal_distribution<float> __dist(0.f, 1.f);

float normalRand()
{
    if(!__gen)
    {
        struct timeval t;
        gettimeofday(&t, nullptr);
        __gen = new std::default_random_engine(t.tv_usec);
    }

    return __dist(*__gen);
}
