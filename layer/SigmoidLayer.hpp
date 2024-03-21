//
// Created by IshimotoKiko on 2024/03/19.
//

#ifndef STUDY_NN_SIGMOIDLAYER_HPP
#define STUDY_NN_SIGMOIDLAYER_HPP
#include "BaseLayer.hpp"

class SigmoidLayer : public BaseLayer{
  Matrix weight_ = Matrix();
  Matrix bias_ = Matrix();
  double learning_rate_ = 0;
public:
  SigmoidLayer(int num);
  Matrix forward_(Matrix in) override;
  Matrix backward_(Matrix grad) override;
  void update_(Matrix grad) override;
};

#endif // STUDY_NN_SIGMOIDLAYER_HPP
