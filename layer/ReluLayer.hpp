//
// Created by IshimotoKiko on 2024/03/22.
//

#ifndef STUDY_NN_RELULAYER_HPP
#define STUDY_NN_RELULAYER_HPP

#include "BaseLayer.hpp"
class ReluLayer : public BaseLayer{
public:
  ReluLayer(int num);
  Matrix forward_(Matrix in) override;
  Matrix backward_(Matrix grad) override;
  void update_(Matrix grad) override;
};

#endif // STUDY_NN_RELULAYER_HPP
