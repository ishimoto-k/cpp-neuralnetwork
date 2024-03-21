//
// Created by IshimotoKiko on 2024/03/19.
//

#include "SigmoidLayer.hpp"


double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x){
  return x * (1.0 - x);
}

SigmoidLayer::SigmoidLayer(int num) {
  input_ = Matrix(0, num);
  output_ = Matrix(0, num);
}
Matrix SigmoidLayer::forward_(Matrix in) {
  return in(sigmoid_);
}
Matrix SigmoidLayer::backward_(Matrix grad) {
  return grad & output_(sigmoid_derivative);
}
void SigmoidLayer::update_(Matrix grad) {

}