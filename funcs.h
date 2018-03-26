#ifndef FUNCS_H
#define FUNCS_H

#include <algorithm>
#include <cmath>
#include <iostream>

#include "type.h"
class Matrix;

void customImpl();

fpt normalRand(fpt n);
fpt sigmoid(fpt z);
fpt sigmoidPrime(fpt z);

void add(fpt_vect& r, const fpt_vect& v);
void sub(fpt_vect& r, const fpt_vect& v);
void dot(fpt_vect& r, const fpt_vect& v);

// Leaves all elements as zero, except the largest
void hardMax(fpt_vect& a);

// Read in a matrix that was printed by python
Matrix pythonRead(std::istream& is);

std::ostream& operator << (std::ostream& os, const fpt_vect& v);

#define assert(x) if(!(x)){ std::cerr << "assert fail: " << (#x) << std::endl; throw std::runtime_error(std::string(#x)); }

#endif // FUNCS_H
