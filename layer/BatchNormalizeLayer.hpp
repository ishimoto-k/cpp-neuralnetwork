//
// Created by IshimotoKiko on 2024/03/21.
//

#ifndef STUDY_NN_BATCHNORMALIZELAYER_HPP
#define STUDY_NN_BATCHNORMALIZELAYER_HPP
#include "BaseLayer.hpp"
class BatchNormalizeLayer : public BaseLayer{
  Matrix gamma = Matrix();
  Matrix beta = Matrix();

  Matrix standard = Matrix();
  Matrix normalized = Matrix();
  Matrix mean_out = Matrix();
  double learning_rate_;
  int batch_size_;
  std::vector<double> normalize(std::vector<double> in, double mu, double ver);
public:
  BatchNormalizeLayer(int num, int batch_size, double learning_rate);
  Matrix forward_(Matrix in) override;
  Matrix backward_(Matrix grad) override;
  void update_(Matrix grad) override;
};

#endif // STUDY_NN_BATCHNORMALIZELAYER_HPP
