#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

template<typename T>
class Network
{
public:
    Network(const std::initializer_list<T>& init)
    {
        layers_.reserve(init.end() - init.begin());

        int prevSize = 0;
        for(auto i = init.begin(); i != init.end(); ++i)
        {
            // No layer for the first (input) layer
            if(prevSize > 0)
            {
                layers_.push_back(Layer<T>(*i, prevSize));
            }
            prevSize = *i;
        }
    }

    std::vector<T> forward(const std::vector<T>& input) const
    {
        auto output = input;
        for(auto i = layers_.begin(); i != layers_.end(); ++i)
        {
            output = i->forward(output);
        }
        return output;
    }

private:
    std::vector<Layer<T>> layers_;
};

#endif // NETWORK_H
