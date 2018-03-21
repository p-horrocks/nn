#ifndef MNIST_H
#define MNIST_H

#include <cstdint>
#include <memory>
#include <vector>

class Mnist
{
public:
    struct Image
    {
        int rows;
        int cols;
        int label;
        std::vector<uint8_t> data;
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
