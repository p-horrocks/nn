cmake_minimum_required(VERSION 2.8)

project(nn)
add_definitions(-std=c++14)
add_executable(${PROJECT_NAME}
    cost.cpp
    funcs.cpp
    layer.cpp
    main.cpp
    matrix.cpp
    mnist.cpp
    network.cpp
    tf.cpp
    )
