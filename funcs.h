#ifndef FUNCS_H
#define FUNCS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "type.h"

fpt normalRand();

fpt sigmoid(fpt z);
void sigmoid(fpt_vect& a);

fpt sigmoidPrime(fpt z);
void sigmoidPrime(fpt_vect& a);

void add(fpt_vect& r, const fpt_vect& v);
void sub(fpt_vect& r, const fpt_vect& v);
void dot(fpt_vect& r, const fpt_vect& v);

// Leaves all elements as zero, except the largest
void hardMax(fpt_vect& a);

std::ostream& operator << (std::ostream& os, const fpt_vect& v);

#endif // FUNCS_H
