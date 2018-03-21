#include "mnist.h"

#include <fstream>

namespace
{

uint32_t read32(std::istream& is)
{
    uint8_t buf[4] = {0};
    is.read(reinterpret_cast<char*>(buf), 4);
    return
            (buf[0] << 24) |
            (buf[1] << 16) |
            (buf[2] << 8) |
            buf[3];
}

} // namespace

Mnist::Mnist()
{
}

int Mnist::loadTrainingData()
{
    return loadData(
                "/home/peter/git/nn/train-images-idx3-ubyte",
                "/home/peter/git/nn/train-labels-idx1-ubyte",
                0,
                50000
                );
}

int Mnist::loadEvaluationData()
{
    return loadData(
                "/home/peter/git/nn/train-images-idx3-ubyte",
                "/home/peter/git/nn/train-labels-idx1-ubyte",
                50000,
                0x0fffffff
                );
}

int Mnist::loadTestData()
{
    return loadData(
                "/home/peter/git/nn/t10k-images-idx3-ubyte",
                "/home/peter/git/nn/t10k-labels-idx1-ubyte",
                0,
                0x0fffffff
                );
}

int Mnist::loadData(
        const char* imageFile,
        const char* labelFile,
        int firstImage,
        int maxImages
        )
{
    std::ifstream img(imageFile);
    if(!img.is_open())
        throw std::runtime_error("cannot open image file");

    std::ifstream lbl(labelFile);
    if(!lbl.is_open())
        throw std::runtime_error("cannot open label file");

    // First 4 bytes is a magic number
    if(read32(img) != 0x00000803)
        throw std::runtime_error("image file had wrong magic");
    if(read32(lbl) != 0x00000801)
        throw std::runtime_error("label file had wrong magic");

    // Next 4 bytes is the image count
    uint32_t n1 = read32(img);
    uint32_t n2 = read32(lbl);
    if(n1 != n2)
        throw std::runtime_error("label and image files had differing image counts");

    // Image file contains the image dimensions next
    uint32_t rows = read32(img);
    uint32_t cols = read32(img);

    // Fast forward if requested
    if(firstImage > 0)
    {
        lbl.seekg(firstImage, std::ios::cur);
        img.seekg(firstImage * rows * cols, std::ios::cur);
    }

    for(int i = 0; i < maxImages; ++i)
    {
        auto image = std::make_shared<Image>();

        uint8_t label;
        lbl.read(reinterpret_cast<char*>(&label), 1);
        image->label = label;
        image->rows  = rows;
        image->cols  = cols;
        image->data.resize(rows * cols);
        img.read(reinterpret_cast<char*>(image->data.data()), image->data.size());

        if((lbl.gcount() == 0) || (img.gcount() == 0) || lbl.eof() || img.eof())
            break;

        images_.push_back(image);
    }

    return images_.size();
}
