cmake_minimum_required(VERSION 3.23)
project(study_nn)

set(CMAKE_CXX_STANDARD 14)

add_executable(study_nn1_simple_perceptron study_nn1_simple_perceptron.cpp)
add_executable(study_nn2_matrix study_nn2_matrix.cpp)

add_executable(study_nn3_simple_perception_backpropagation study_nn3_simple_perception_backpropagation.cpp)

add_executable(study_nn4_multi_perceptron_backpropagation
        study_nn4_multi_perceptron_backpropagation.cpp
        matrix/Matrix.hpp)

add_executable(study_nn5_mnist
        study_nn5_mnist.cpp
        matrix/Matrix.hpp)
#target_link_libraries(study_nn5_mnist libpng)
