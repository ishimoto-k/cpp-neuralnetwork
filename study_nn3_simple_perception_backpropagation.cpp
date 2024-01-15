//
// Created by IshimotoKiko on 2023/12/02.
//
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}
double sigmoid_derivative(double x){
  return x * (1.0 - x);
}

double pro(std::vector<double> weights, std::vector<double> inputs){
  double ans = 0;
  for(int i = 0; i < weights.size(); i++){
    ans += weights[i] * inputs[i];
  }
  return ans;
}
int main() {
#define INPUT_NUM 2
  const double bias = 1.0;
  const double learning_rate = 0.1;
  std::vector<double> weights;
  for(int i = 0;i < INPUT_NUM+1; i++){
    weights.push_back((double)rand() / RAND_MAX);
  }
  std::function<double(double)> activation_function = sigmoid_;
  std::function<double(double)> activation_function_derivative = sigmoid_derivative;

  //training
  for(int epoch = 0; epoch < 10000; epoch++){
    const std::vector<std::vector<double>> teacher_input = {{0,0},{0,1},{1,0},{1,1},};
    const std::vector<double> teacher_output = {1,1,1,0};

    for(int pattern_i = 0; pattern_i < teacher_input.size(); pattern_i++){
      std::vector<double> inputs = teacher_input[pattern_i];
      inputs.push_back(bias);
      const double teacher = teacher_output[pattern_i];

      //forward
      double output = activation_function(pro(weights, inputs));

      //backward
      double error = teacher - output;
      for(int i = 0; i < weights.size(); i++){
        weights[i] += learning_rate * error * activation_function_derivative(output) * inputs[i];
      }
    }
  }



  //input to trained model
  while (true) {
    std::vector<double> inputs = {};
    for (int i = 0; i < INPUT_NUM; i++) {
      double in;
      std::cout << "input_" << i << ": ";
      std::cin >> in;
      inputs.push_back(in);
    }
    inputs.push_back(bias);

    std::cout << "output: " << activation_function(pro(weights, inputs)) << std::endl;
  }
}
