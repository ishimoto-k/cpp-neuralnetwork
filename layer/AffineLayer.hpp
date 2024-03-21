//
// Created by IshimotoKiko on 2024/03/19.
//

#ifndef STUDY_NN_AFFINELAYER_HPP
#define STUDY_NN_AFFINELAYER_HPP
#include "BaseLayer.hpp"

class AffineLayer : public BaseLayer{
  Matrix weight_ = Matrix();
  Matrix bias_ = Matrix();
  double learning_rate_ = 0;
public:
  AffineLayer(int input_num, int output_num, double learning_rate);
  Matrix forward_(Matrix in) override;
  Matrix backward_(Matrix grad) override;
  void update_(Matrix grad) override;
};

#endif // STUDY_NN_AFFINELAYER_HPP
