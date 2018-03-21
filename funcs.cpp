#include "funcs.h"

#include <random>

std::default_random_engine __gen;
std::normal_distribution<float> __dist(0.f, 1.f);

float normalRand()
{
    return __dist(__gen);
}
