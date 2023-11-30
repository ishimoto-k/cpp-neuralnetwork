#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}
double tanh_(double x){
  return std::tanh(x);
}
double ReLU_(double x){
  return std::max(x, 0.0);
}
int main() {
  #define INPUT_NUM 2
  const double bias = 1.0;
  std::vector<double> weight = {-10,-10,15};
  std::vector<double> inputs = {};
  std::function<double(double)> activation_function = sigmoid_;

  while (true) {
    for (int i = 0; i < INPUT_NUM; i++) {
      double in;
      std::cout << "input_" << i << ": ";
      std::cin >> in;
      inputs.push_back(in);
    }
    inputs.push_back(bias);

    double z = 0;
    for (int i = 0; i < weight.size(); i++) {
      z += weight[i] * inputs[i];
    }

    std::cout << "output: " << activation_function(z) << std::endl;
    inputs.clear();
  }
}
