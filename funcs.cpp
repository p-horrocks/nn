#include "funcs.h"

#include <random>
#include <sstream>
#include <sys/time.h>

#include "mnist.h"
#include "network.h"

std::default_random_engine* __gen = nullptr;

void customImpl()
{
    Mnist train;
    int n1 = train.loadTrainingData();

    Mnist eval;
    int n2 = eval.loadEvaluationData();

    Mnist test;
    int n3 = test.loadTestData();

    std::cout << "loaded " << n1 << " training, " << n2 << " eval and " << n3 << " test images" << std::endl;

    // Single layer NN to take in the MNIST images and generate 10 outputs -
    // the probability that each image is a corresponding digit
    Network net({784, 30, 10}, std::make_shared<CrossEntropyCost>());

    const int epochs    = 30;
    const int batchSize = 10;
    const fpt learnRate = .5f;
    const fpt lambda    = 5.f;
    net.trainMNIST_SGD(epochs, batchSize, learnRate, lambda, train, test);
}

fpt normalRand(fpt n)
{
    if(!__gen)
    {
        struct timeval t;
        gettimeofday(&t, nullptr);
        __gen = new std::default_random_engine(t.tv_usec);
    }

    std::normal_distribution<fpt> d(0.f, 1.f / std::sqrt(n));
    return d(*__gen);
}

fpt sigmoid(fpt z)
{
    return (fpt)1 / ((fpt)1 + std::exp(-z));
}

fpt sigmoidPrime(fpt z)
{
    fpt s = sigmoid(z);
    return s * ((fpt)1 - s);
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

Matrix pythonRead(std::istream& is)
{
    std::locale loc;
    Matrix retval;

    // Reads until the given character is found and checks that only whitespace
    // was read to the charactger
    auto nextNonWhite = [&](std::istream& is){
        char ch;
        is.get(ch);
        while(is.good() && std::isspace(ch, loc))
        {
            is.get(ch);
        }
        return ch;
    };

    char ch = nextNonWhite(is);
    if(ch != '[')
        throw std::runtime_error("matrix start not found");

    for(;;)
    {
        char ch = nextNonWhite(is);
        if(ch == ']')
            break;

        if(ch != '[')
            throw std::runtime_error("row start not found");

        std::string line;
        std::getline(is, line, ']');
        // Add a space at the end. Without the space it's impossible to tell
        // the difference between a successful extraction that finished due to
        // EOF, and a failed extraction because no characters were extracted
        // before the EOF. ie: " 0.1]" and " ]" both look valid
        line += " ";

        fpt_vect row;
        std::istringstream ls(line);
        fpt v;
        while(!ls.eof())
        {
            ls >> v;
            if(ls.good())
            {
                row.push_back(v);
            }
        }

        if(retval.cols() == 0)
        {
            retval.resize(0, row.size(), false);
            retval.appendRow(row);
        }
        else if(retval.cols() != row.size())
        {
            throw std::runtime_error("incorrect row length");
        }
        else
        {
            retval.appendRow(row);
        }
    }

    return retval;
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
