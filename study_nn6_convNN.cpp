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
double relu_derivative_(double x){
  //  std::cout << "relu_derivative d:" << (*matrix)[col][row] << " x:" << x << " " << ((*matrix)[col][row] > 0 ? x : 0) << std::endl;
  return x > 0 ? 1 : 0;
}
double relu_derivative(double x, Matrix* matrix, int col, int row){
//  std::cout << "relu_derivative d:" << (*matrix)[col][row] << " x:" << x << " " << ((*matrix)[col][row] > 0 ? x : 0) << std::endl;
  return (*matrix)[col][row] > 0 ? x : 0;
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
  int count = 0;
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
    for(int s=0; s<10; s++){
      if(s==t) {
        teacher.push_back(1);
      }else {
        teacher.push_back(0);
      }
    }
    teacher_data.push_back(teacher);
    data.push_back(tmp);
    count++;
    if(count==10000){
      break;
    }
  }

  return {data, teacher_data};
}

struct BatchParameter{

  std::vector<double> gamma;
  std::vector<double> beta;
  double epsilon;
};
#include <sstream>
int main() {
  std::vector<Matrix> src = {
    std::vector<std::vector<double>>{{0,0,1}, {0,0,2}, {0,0,3}},
    std::vector<std::vector<double>>{{0,0,4}, {0,0,5}, {0,0,6}},
  };
  for(int i = 0; i<src.size(); i++){
    src[i].printMatrix();
  }
  auto mat = Matrix::Flatten(src);
  mat.printMatrix();

  auto ret = Matrix::FromFlatten(3,3,mat[0],2);
  for(int i = 0; i<ret.size(); i++){
    ret[i].printMatrix();
  }
#define THRESHOLD 1000
#define INPUT_NUM 1
#define MIDDLE_NUM 100
#define OUTPUT_NUM 10
#define WIDTH 28
#define HEIGHT 28
#define KERNEL_SIZE 5
#define KERNEL_STRIDE 1
#define CONV_OUTPUT ((HEIGHT - KERNEL_SIZE) / KERNEL_STRIDE + 1)
#define POOLING_SIZE 2
#define POOLING_STRIDE 2
#define PADDING 0
#define POOLING_OUTPUT (((CONV_OUTPUT - POOLING_SIZE) / POOLING_STRIDE + 1) + PADDING * 2)
#define FILTER_NUM 5

#define HEIGHT2 POOLING_OUTPUT
#define KERNEL_SIZE2 5
#define KERNEL_STRIDE2 1
#define CONV_OUTPUT2 ((HEIGHT2 - KERNEL_SIZE2) / KERNEL_STRIDE2 + 1)
#define POOLING_SIZE2 2
#define POOLING_STRIDE2 2
#define PADDING2 0
#define POOLING_OUTPUT2 (((CONV_OUTPUT2 - POOLING_SIZE2) / POOLING_STRIDE2 + 1) + PADDING2 * 2)
#define FILTER_NUM2 30
  std::cout << "POOLING_OUTPUT " << POOLING_OUTPUT << std::endl;
  std::cout << "POOLING_OUTPUT2 " << POOLING_OUTPUT2 << std::endl;
  auto r = read_csv(MNIST_TRAIN_PATH);
  const double conv_learning_rate = 0.001;
  const double nn_learning_rate = 0.01;
  Matrix middle_weights = Matrix::Random(MIDDLE_NUM, POOLING_OUTPUT2*POOLING_OUTPUT2*FILTER_NUM2);
//  std::cout << "middle_weights"; middle_weights.printMatrix();
  Matrix middle_bias_weights = Matrix::Random(MIDDLE_NUM, 1);
  Matrix output_weights = Matrix::Random(OUTPUT_NUM, MIDDLE_NUM);
  Matrix output_bias_weights = Matrix::Random(OUTPUT_NUM, 1);

  typedef std::vector<Matrix> Kernel;
  std::vector<Kernel> kernels;
  std::vector<Matrix> conv_bias_weight;
  std::vector<Kernel> kernels2;
  std::vector<Matrix> conv_bias_weight2;
  for(int i=0; i<FILTER_NUM; i++){
    Kernel kernel;
    for(int j=0; j<INPUT_NUM; j++){
      kernel.push_back(Matrix::RandomXavier(KERNEL_SIZE, KERNEL_SIZE));
    }
    kernels.push_back(kernel);
    //    filter[i].printMatrix();
    conv_bias_weight.push_back(Matrix::Random(1,1));
  }
  for(int i=0; i<FILTER_NUM2; i++){
    Kernel kernel;
    for(int j=0; j<FILTER_NUM; j++){
      kernel.push_back(Matrix::RandomXavier(KERNEL_SIZE2, KERNEL_SIZE2));
    }
    kernels2.push_back(kernel);
    //    filter[i].printMatrix();
    conv_bias_weight2.push_back(Matrix::Random(1,1));
  }

  //training
  for(int epoch = 0; epoch < r.first.size(); epoch++){
    std::cout << epoch << "/" << r.first.size() << std::endl;
    Matrix teacher_input = std::vector<std::vector<double>>{r.first[epoch]};
    Matrix teacher_output = std::vector<std::vector<double>>{r.second[epoch]};

    for(int pattern_i = 0; pattern_i < teacher_input.cols(); pattern_i++){
      std::cout << "size " << teacher_input[pattern_i].size() << std::endl;
      std::vector<Matrix> inputs = {Matrix(WIDTH, HEIGHT, teacher_input[pattern_i])};
//      inputs.printMatrix();
      Matrix teacher = Matrix(std::vector<std::vector<double>>{teacher_output[pattern_i]});
//      std::cout << "teacher"; teacher.printMatrix();

      //forward
      std::vector<Matrix> convolution;
      std::vector<Matrix> relued;
      std::vector<Matrix> pooled;
      // convolution
      for(int i=0; i<FILTER_NUM; i++){
        Matrix matrix(CONV_OUTPUT, CONV_OUTPUT);
        for(int j=0; j<INPUT_NUM; j++){
          matrix = matrix + inputs[j].convolution(kernels[i][j], KERNEL_STRIDE);
        }
        // convolution
        convolution.push_back(matrix + conv_bias_weight[i]);
      }
      for(int i=0; i<FILTER_NUM; i++){
        // relued
        relued.push_back(convolution[i](relu_));
      }
      for(int i=0; i<FILTER_NUM; i++){
        // pooled
        pooled.push_back(Matrix::pooling(relued[i], POOLING_SIZE, POOLING_STRIDE).padding(PADDING));
      }

      std::vector<Matrix> convolution2;
      std::vector<Matrix> relued2;
      std::vector<Matrix> pooled2;
      // convolution
      for (int i = 0; i < FILTER_NUM2; i++) {
        Matrix matrix(CONV_OUTPUT2, CONV_OUTPUT2);
        for(int j=0; j<FILTER_NUM; j++){
          matrix = matrix + pooled[j].convolution(kernels2[i][j], KERNEL_STRIDE2);
        }
        // convolution
        convolution2.push_back(matrix + conv_bias_weight2[i]);
      }
      for (int i = 0; i < FILTER_NUM2; i++) {
        // relued
        relued2.push_back(convolution2[i](relu_));
      }
      for (int i = 0; i < FILTER_NUM2; i++) {
        // pooled
        pooled2.push_back(Matrix::pooling(relued2[i], POOLING_SIZE2, POOLING_STRIDE2));
      }

      // flatten
      std::cout << "pooled2 size = " << pooled2.size() <<std::endl;
      auto flatten = Matrix::Flatten(pooled2);
      flatten.printSize();
      // affine1
      auto middle_sum = (middle_weights * flatten.t() + middle_bias_weights);
//      auto middle_output = middle_sum(sigmoid_);
      auto middle_output = middle_sum(relu_);
      // affine2
      auto output_sum = (output_weights * middle_output + output_bias_weights);
//      auto output = output_sum(sigmoid_);
      auto y = Matrix::softmax(output_sum.t()).t();
      auto loss = Matrix::cross_entropy_error(y.t(), teacher);
//      auto output = Matrix::softmax_forward(output_sum.t()).t();
      std::cout << "output"; y.t().printMatrix();
      std::cout << "teacher"; teacher.printMatrix();
      std::cout << "loss " << loss << std::endl;

      //backward
      // affine1
      //activate
      auto output_error = Matrix::SoftmaxWithLossBackward(output_sum.t(), teacher).t();
//      auto output_error = Matrix::softmax_backward(teacher, output.t()).t();
//      auto output_error = (output - teacher.t()) & output(sigmoid_derivative);
      //affine
      output_weights = output_weights - nn_learning_rate * output_error * middle_output.t();
      output_bias_weights = output_bias_weights - nn_learning_rate * output_error;
      // affine2
      auto middle_error = (output_error.t() * output_weights);
//      middle_error(relu_derivative, &middle_output);
      auto delta = middle_error(relu_derivative, &middle_output).t();
      auto bias_middle_delta = nn_learning_rate * Matrix::clip_gradients(delta, THRESHOLD);
//      auto bias_middle_delta = nn_learning_rate * (middle_error.t() & middle_output(sigmoid_derivative));
      middle_weights = middle_weights - (bias_middle_delta * flatten);
      middle_bias_weights = middle_bias_weights - bias_middle_delta;
      auto grad_error = (middle_error * middle_weights);


//      std::cout << "grad_error"; grad_error.printMatrix();
      //flatten
      auto grads = Matrix::FromFlatten(pooled2[0].cols(), pooled2[0].rows(), grad_error[0], FILTER_NUM2);

      std::vector<Matrix> diff_x = std::vector<Matrix>(FILTER_NUM, Matrix(pooled[0].cols(),pooled[0].rows()));


      //convolution2 backward
      for (int channel = 0; channel < FILTER_NUM2; channel++) {
        auto &grad = grads[channel];
        auto grad_pooling = Matrix::grad_pooling(grad, relued2[channel], POOLING_SIZE2, POOLING_STRIDE2);
        grad_pooling = grad_pooling(relu_derivative, &convolution2[channel]);
        grads[channel] = grad_pooling;
      }
      for (int out = 0; out < FILTER_NUM; out++) {
        for (int channel = 0; channel < FILTER_NUM2; channel++) {
          int& kernel = channel;
          auto rotated_f = kernels2[kernel][out].rotate180();
          auto grad = rotated_f.padding(grads[channel].cols() - 1).convolution(grads[channel], KERNEL_STRIDE2);
//          auto grad = grads[channel].padding(rotated_f.cols() - 1).convolution(rotated_f, KERNEL_STRIDE2);
          diff_x[out] = diff_x[out] + grad;
        }
      }
      //  convolution2 filter update
      for (int kernel = 0; kernel < FILTER_NUM2; kernel++){
        for (int input = 0; input < FILTER_NUM; input++) {
          int& filter = input;
          auto forward_input = pooled[input];
//          auto filter_error = forward_input.convolution(grads[kernel], KERNEL_SIZE2);
          auto filter_error = forward_input.convolution(grads[kernel], KERNEL_SIZE2);
          kernels2[kernel][filter] = kernels2[kernel][filter] - conv_learning_rate * Matrix::clip_gradients(filter_error, THRESHOLD);
        }
        conv_bias_weight2[kernel] = conv_bias_weight2[kernel] - conv_learning_rate * grads[kernel].sum();
      }

      //convolution backward
      for (int channel = 0; channel < FILTER_NUM; channel++) {
        auto &grad = diff_x[channel];
        auto grad_pooling = Matrix::grad_pooling(grad, relued[channel], POOLING_SIZE, POOLING_STRIDE);
        grad_pooling = grad_pooling(relu_derivative, &convolution[channel]);
        diff_x[channel] = grad_pooling;
      }
      //  convolution filter update
      for (int kernel = 0; kernel < FILTER_NUM; kernel++){
        for (int input = 0; input < INPUT_NUM; input++) {
          int filter = input;
          auto forward_input = inputs[input];
          auto filter_error = forward_input.convolution(diff_x[kernel], KERNEL_SIZE);
          kernels[kernel][filter] = kernels[kernel][filter] - conv_learning_rate * Matrix::clip_gradients(filter_error, THRESHOLD);
        }
        conv_bias_weight[kernel] = conv_bias_weight[kernel] - conv_learning_rate * diff_x[kernel].sum();
      }
    }
  }
  return 0;

}