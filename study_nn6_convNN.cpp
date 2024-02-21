//
// Created by IshimotoKiko on 2023/12/28.
//

#include "matrix/Matrix.hpp"
#include <fstream>
#include <regex>
#include <sstream>

double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}
double sigmoid_derivative(double x){
  return x * (1.0 - x);
}

double relu_(double x){
  return std::max(0.0, x);
}
double relu_derivative(double x){
  return x > 0 ? 1 : 0;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_csv(const std::string& file_name) {
  std::vector<std::vector<double>> data;
  std::vector<std::vector<double>> teacher_data;

  std::ifstream file(file_name);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << file_name << std::endl;
    exit(0);
  }

  int t = 0;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;
    int i = 0;
    std::vector<double> tmp;
    while (std::getline(ss, token, ',')) {
      try {
        if(i==0){
          t = std::stoi(token);
        }else {
          double value = std::stod(token)/255;
          tmp.push_back(value);
        }
      } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
      } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
      }
      i++;
    }

    std::vector<double> teacher;
    for(int i=0; i<10; i++){
      if(i==t) {
        teacher.push_back(1);
      }else {
        teacher.push_back(0);
      }
    }
    teacher_data.push_back(teacher);
    data.push_back(tmp);
//    break;
  }

  return {data, teacher_data};
}


int main() try {
#define MIDDLE_NUM 100
#define OUTPUT_NUM 10
#define WIDTH 28
#define HEIGHT 28
#define FILTER_SIZE 3
#define FILTER_STRIDE 1
#define CONV_OUTPUT ((HEIGHT - FILTER_SIZE) / FILTER_STRIDE + 1)
#define POOLING_SIZE 2
#define POOLING_STRIDE 2
#define POOLING_OUTPUT ((CONV_OUTPUT - POOLING_SIZE) / POOLING_STRIDE + 1)
  std::cout << "POOLING_OUTPUT" << POOLING_OUTPUT << std::endl;
  auto r = read_csv("mnist_train.csv");
  const double conv_learning_rate = 0.11;
  const double nn_learning_rate = 0.01;
  Matrix middle_weights = Matrix::Random(MIDDLE_NUM, POOLING_OUTPUT*POOLING_OUTPUT);
  std::cout << "middle_weights"; middle_weights.printMatrix();
  Matrix middle_bias_weights = Matrix::Random(MIDDLE_NUM, 1);
  Matrix output_weights = Matrix::Random(OUTPUT_NUM, MIDDLE_NUM);
  Matrix output_bias_weights = Matrix::Random(OUTPUT_NUM, 1);

  Matrix filter = Matrix::Random(FILTER_SIZE);
  Matrix conv_bias_weight = Matrix::Random(1,1);

  std::function<double(double)> activation_function = sigmoid_;
  std::function<double(double)> activation_function_derivative = sigmoid_derivative;
  //training
  for(int epoch = 0; epoch < r.first.size(); epoch++){
    std::cout << epoch << "/" << r.first.size() << std::endl;
    Matrix teacher_input = std::vector<std::vector<double>>{r.first[epoch]};
    Matrix teacher_output = std::vector<std::vector<double>>{r.second[epoch]};

    for(int pattern_i = 0; pattern_i < teacher_input.cols(); pattern_i++){
      std::cout << "size " << teacher_input[pattern_i].size() << std::endl;
      Matrix inputs = Matrix(WIDTH, HEIGHT, teacher_input[pattern_i]);
      Matrix teacher = Matrix(std::vector<std::vector<double>>{teacher_output[pattern_i]});
      std::cout << "teacher"; teacher.printMatrix();

      //forward
      // convolution
      auto convoluted = (inputs.convolution(filter, FILTER_STRIDE) + conv_bias_weight);
//      std::cout << "convoluted"; convoluted.printMatrix();

      // pooling
      auto pooled = Matrix::pooling(convoluted, POOLING_SIZE, POOLING_STRIDE);
//      std::cout << "pooled"; pooled.printMatrix();
      // flatten
      auto flatten = Matrix::Flatten(pooled);
//      std::cout << "flatten"; flatten.printMatrix();

      // affine
      auto middle_output = (middle_weights * flatten.t() + middle_bias_weights)(sigmoid_);
      auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
//      std::cout << "middle_output"; middle_output.printMatrix();
      std::cout << "output"; output.t().printMatrix();


      //backward
      // affine
      auto output_error = (output - teacher.t()) & output(sigmoid_derivative);
      auto bias_output_delta = nn_learning_rate * output_error;
      output_weights = output_weights - bias_output_delta * middle_output.t();
      output_bias_weights = output_bias_weights - bias_output_delta;

      auto middle_error = (output_error.t() * output_weights);
      auto bias_middle_delta = nn_learning_rate * (middle_error.t() & middle_output(sigmoid_derivative));
      middle_weights = middle_weights - (bias_middle_delta * flatten);
//      std::cout << "middle_weights"; middle_weights.printMatrix();
      middle_bias_weights = middle_bias_weights - bias_middle_delta;

      //flatten
      auto grad = Matrix(pooled.cols(), pooled.rows(), middle_error[0]);
//      std::cout << "grad"; grad.printMatrix();

      //pooling
      auto grad_pooling = Matrix::grad_pooling(convoluted, grad, POOLING_SIZE, POOLING_STRIDE);
//      std::cout << "grad_pooling"; grad_pooling.printMatrix();

      //convolution
      auto dfilter = Matrix::backward_convolution(inputs, grad_pooling, filter, FILTER_STRIDE);
//      std::cout << "dfilter"; dfilter.printMatrix();
      filter = filter - conv_learning_rate * dfilter;
      conv_bias_weight = conv_bias_weight - conv_learning_rate * dfilter.average();
//      std::cout << "filter"; filter.printMatrix();
    }
  }




  auto test = read_csv("mnist_test.csv");
  for (int i=0; i<test.first.size(); i++){
    std::cout << i << "/" << test.first.size() << std::endl;

    Matrix inputs = std::vector<std::vector<double>> {test.first[i]};
    Matrix answer = std::vector<std::vector<double>> {test.second[i]};

    //forward

    auto convoluted = (inputs.convolution(filter, FILTER_STRIDE) + conv_bias_weight);
    //      std::cout << "convoluted"; convoluted.printMatrix();

    // pooling
    auto pooled = Matrix::pooling(convoluted, POOLING_SIZE, POOLING_STRIDE);
    //      std::cout << "pooled"; pooled.printMatrix();
    // flatten
    auto flatten = Matrix::Flatten(pooled);
    //      std::cout << "flatten"; flatten.printMatrix();

    // affine
    auto middle_output = (middle_weights * flatten.t() + middle_bias_weights)(sigmoid_);
    auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
    std::cout << "answer"; answer.printMatrix();
    std::cout << "output"; output.t().printMatrix();

  }
} catch(MatrixSizeMismatchException e){
  std::cerr << e.what() << std::endl;
}
