//
// Created by IshimotoKiko on 2023/12/28.
//

#include "matrix/Matrix.hpp"


double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}
double sigmoid_derivative(double x){
  return x * (1.0 - x);
}

int main() {
#define INPUT_NUM 2
#define MIDDLE_NUM 4
#define OUTPUT_NUM 1
  const double learning_rate = 1;
  Matrix middle_weights = Matrix::Random(MIDDLE_NUM, INPUT_NUM);
  Matrix middle_bias_weights = Matrix::Create(MIDDLE_NUM, 1, 1);
  Matrix output_weights = Matrix::Random(OUTPUT_NUM, MIDDLE_NUM);
  Matrix output_bias_weights = Matrix::Create(OUTPUT_NUM, 1, 1);
  std::cout << "middle_weights"; middle_weights.printMatrix();
  std::cout << "output_weights"; output_weights.printMatrix();
  std::function<double(double)> activation_function = sigmoid_;
  std::function<double(double)> activation_function_derivative = sigmoid_derivative;
  //training
  for(int epoch = 0; epoch < 1000; epoch++){
    Matrix teacher_input = std::vector<std::vector<double>>{{0,0},{0,1},{1,0},{1,1},};
    const std::vector<double> teacher_output = {1,0,0,1};

    for(int pattern_i = 0; pattern_i < teacher_input.cols(); pattern_i++){
      Matrix inputs = std::vector<std::vector<double>> {teacher_input[pattern_i]};
      std::cout << "inputs"; inputs.printMatrix();
      const double teacher = teacher_output[pattern_i];

      //forward
      std::cout << "forward" << std::endl;
      auto middle_output = (middle_weights * inputs.t() + middle_bias_weights)(sigmoid_);
      auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
      std::cout << "output"; output.printMatrix();


      //backward
      std::cout << "backward" << std::endl;
      auto output_error = (Matrix({{teacher}}) - output) & output(sigmoid_derivative);
      auto middle_error = (output_error * output_weights.t());
      auto bias_output_delta = learning_rate * output_error;
      output_weights = output_weights + bias_output_delta * middle_output.t();
      output_bias_weights = output_bias_weights + bias_output_delta;
      std::cout << "output_weights"; output_weights.printMatrix();


      std::cout << std::endl << "update middle_weights"  << std::endl;
      auto bias_middle_delta = learning_rate * (middle_error & middle_output(sigmoid_derivative));
      middle_weights = middle_weights + bias_middle_delta * inputs;
      middle_bias_weights = middle_bias_weights + bias_middle_delta;
      std::cout << "middle_weights"; middle_weights.printMatrix();
      std::cout << "end" << std::endl << std::endl;
    }
  }



  //input to trained model
  while (true) {

    Matrix inputs = std::vector<std::vector<double>> {{}};
    for (int i = 0; i < INPUT_NUM; i++) {
      double in;
      std::cout << "input_" << i << ": ";
      std::cin >> in;
      inputs[0].push_back(in);
    }

    //forward
    auto middle_output = (middle_weights * inputs.t() + middle_bias_weights)(sigmoid_);
    std::cout << "middle_output"; middle_output.printMatrix();
    auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
    std::cout << "output"; output.printMatrix();
  }
}
