#ifndef MNIST_H
#define MNIST_H

#include <cstdint>
#include <memory>
#include <vector>

#include "type.h"

class Mnist
{
public:
    struct Image
    {
        int rows;
        int cols;
        fpt_vect data;
        fpt_vect label;
    };
    typedef std::shared_ptr<Image> ImagePtr;

    Mnist();

    int loadTrainingData();
    int loadEvaluationData();
    int loadTestData();

    const std::vector<ImagePtr>& images() const { return images_; }

private:
    int loadData(
            const char* imageFile,
            const char* labelFile,
            int firstImage,
            int maxImages
            );

    std::vector<ImagePtr> images_;
};

#endif // MNIST_H
