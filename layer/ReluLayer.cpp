//
// Created by IshimotoKiko on 2024/03/22.
//

#include "ReluLayer.hpp"

double relu(double x){
  return x > 0 ? x : 0;
}
ReluLayer::ReluLayer(int num){
}
Matrix ReluLayer::forward_(Matrix in){
  return in(relu);
}
Matrix ReluLayer::backward_(Matrix grad){
  Matrix dx = grad;
  for(int i = 0; i<grad.cols(); i++){
    for(int j = 0; j<grad.rows(); j++){
      if(output_[i][j]==0){
        dx[i][j] = 0;
      }
    }
  }
  return dx;
}
void ReluLayer::update_(Matrix grad){

}