cmake_minimum_required(VERSION 3.23)
project(study_nn)

if(DEFINED MNIST_TRAIN_PATH)
else()
    set(MNIST_TRAIN_PATH \"../mnist_train.csv\")
endif()
message(MNIST_TRAIN_PATH = ${MNIST_TRAIN_PATH})
add_compile_definitions(MNIST_TRAIN_PATH=\"${MNIST_TRAIN_PATH}\")

if(DEFINED MNIST_TEST_PATH)
else()
    set(MNIST_TEST_PATH \"../mnist_test.csv\")
endif()
message(MNIST_TEST_PATH = ${MNIST_TEST_PATH})
add_compile_definitions(MNIST_TEST_PATH=\"${MNIST_TEST_PATH}\")

set(CMAKE_CXX_STANDARD 14)

add_library(layer STATIC
        layer/AffineLayer.cpp
        layer/SigmoidLayer.cpp)
target_include_directories(layer INTERFACE layer)

add_executable(study_nn1_simple_perceptron study_nn1_simple_perceptron.cpp)
add_executable(study_nn2_matrix study_nn2_matrix.cpp)

add_executable(study_nn3_simple_perception_backpropagation study_nn3_simple_perception_backpropagation.cpp)

add_executable(study_nn4_multi_perceptron_backpropagation
        study_nn4_multi_perceptron_backpropagation.cpp
        matrix/Matrix.hpp)

add_executable(study_nn5_mnist
        study_nn5_mnist.cpp
        matrix/Matrix.hpp)


add_executable(study_nn6_convNN
        study_nn6_convNN.cpp
        matrix/Matrix.hpp)
#target_link_libraries(study_nn5_mnist libpng)
