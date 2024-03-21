//
// Created by IshimotoKiko on 2024/03/19.
//

#include "AffineLayer.hpp"


AffineLayer::AffineLayer(int input_num, int output_num, double learning_rate) {
  input_ = Matrix(1, input_num);
  output_ = Matrix(1, output_num);
  grad_ = input_;
  weight_ = Matrix::Random(input_num, output_num);
  bias_ = Matrix::Random(1, output_num);
  learning_rate_ = learning_rate;
}
Matrix AffineLayer::forward_(Matrix in) {
  return in * weight_ + bias_;
}
Matrix AffineLayer::backward_(Matrix grad) {
  return grad * weight_.t();
}
void AffineLayer::update_(Matrix grad) {
    auto d_weight_ = input_.t() * grad;
    weight_ = weight_ - learning_rate_ * d_weight_;
    bias_ = bias_ - learning_rate_ * grad.sum(COL);
}