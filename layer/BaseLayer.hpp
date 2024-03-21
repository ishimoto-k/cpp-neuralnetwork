//
// Created by IshimotoKiko on 2024/03/19.
//

#ifndef STUDY_NN_BASE_LAYER_HPP
#define STUDY_NN_BASE_LAYER_HPP

#include "../matrix/Matrix.hpp"
class BaseLayer {
protected:
  Matrix input_ = Matrix();
  Matrix output_ = Matrix();
  Matrix grad_ = Matrix();
  virtual Matrix forward_(Matrix in) = 0;
  virtual Matrix backward_(Matrix grad) = 0;
  virtual void update_(Matrix grad) = 0;

public:

  Matrix forward(const Matrix& in){
    input_ = in;
    output_ = forward_(in);
    return output_;
  }
  Matrix backward(const Matrix& grad){
    grad_ = grad;
    return backward_(grad);
  }
  void update(){
    update_(grad_);
  }
};

#endif // STUDY_NN_BASE_LAYER_HPP
