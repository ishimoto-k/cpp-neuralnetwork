//
// Created by IshimotoKiko on 2024/03/21.
//

#include "BatchNormalizeLayer.hpp"

double square(double in){
  return in * in;
}
double sqrt_(double in){
  return std::sqrt(in);
}

double division(double x, Matrix* matrix, int col, int row){
  return x / (*matrix)[col][row];
}

std::vector<double> BatchNormalizeLayer::normalize(std::vector<double> in, double mu, double ver){
  std::vector<double> hat_x = std::vector<double>(in.size());
  for(int i = 0; i<in.size(); i++){
    hat_x[i] = (in[i] - mu) / sqrt(ver + 10e-7);
  }
  return hat_x;
}
BatchNormalizeLayer::BatchNormalizeLayer(int num, int batch_size, double learning_rate){
  input_ = Matrix(batch_size, num);
  output_ = Matrix(batch_size, num);
  gamma = Matrix::Random(1, num);
  beta = Matrix::Random(1, num);
  standard = Matrix(batch_size, 1);
  learning_rate_ = learning_rate;
  batch_size_ = batch_size;
}

Matrix BatchNormalizeLayer::forward_(Matrix in){
  auto mu = in.mean(VECTOR::COL);
  mean_out = in - mu;
  auto variance = mean_out(square).mean(VECTOR::COL);
  standard = (variance + 10e-7)(sqrt_);
  normalized = mean_out/standard;
  auto ret = (normalized & gamma) + beta;
  return ret;
}
Matrix BatchNormalizeLayer::backward_(Matrix grad){
  auto grad_normalized = grad & gamma;
  auto grad_mean_out = grad_normalized / standard;
  auto squared_standard = standard(sqrt_);
  auto grad_standard = ((grad_normalized & mean_out)/squared_standard).sum(VECTOR::COL);
  auto grad_variance = (0.5 * grad_standard) / standard;
  grad_mean_out = grad_mean_out + (2.0/batch_size_ * mean_out) & grad_variance;
  auto grad_mu = grad_mean_out.sum(VECTOR::COL);
  auto dx = (grad_mean_out - grad_mu) * (1.0/batch_size_);

  return dx;
}
void BatchNormalizeLayer::update_(Matrix grad){
  auto dbeta = grad.sum(VECTOR::COL);
  auto dganma = (normalized & grad).sum(VECTOR::COL);
}