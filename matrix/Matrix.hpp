//
// Created by IshimotoKiko on 2023/12/28.
//

#ifndef STUDY_NN_MATRIX_HPP
#define STUDY_NN_MATRIX_HPP
#include <iostream>
#include <utility>
#include <vector>
#include <functional>
#include <cmath>
#include <random>

class Matrix{
  std::vector<std::vector<double>> matrix_ = {};
public:
  static Matrix Random(std::size_t cols, std::size_t rows){
    // 乱数エンジンの生成
    std::random_device rd;
    std::mt19937 gen(rd());

    // 範囲指定（0.0から1.0までの実数）
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> e;
    for(int i = 0; i<cols; i++){
      std::vector<double> vector;
      for(int j = 0; j < rows; j++){
        vector.push_back(dist(gen));
      }
      e.push_back(vector);
    }
    return {e};
  }
  static Matrix Create(std::size_t cols, std::size_t rows, double value){
    std::vector<std::vector<double>> e;
    for(int i = 0; i<cols; i++){
      std::vector<double> vector;
      for(int j = 0; j < rows; j++){
        vector.push_back(value);
      }
      e.push_back(vector);
    }
    return {e};
  }
  static Matrix E(int size){
    std::vector<std::vector<double>> e;
    for(int i = 0; i<size; i++){
      std::vector<double> vector;
      for(int j = 0; j < size; j++){
        if(i == j){
          vector.push_back(1);
        }else{
          vector.push_back(0);
        }
      }
      e.push_back(vector);
    }
    return {e};
  }
  virtual ~Matrix(){

  }

  Matrix(std::size_t cols, std::size_t rows) : matrix_(std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0))) {}
  Matrix(std::vector<std::vector<double>>  mat) : matrix_(std::move(mat)) {}


  double dot(std::vector<double> x, std::vector<double> y){
    double z = 0;
    for(int i=0;i<x.size();i++){
      z += x[i] * y[i];
    }
    return z;
  }
  std::vector<double> column(int n) const{
    std::vector<double> y(matrix_.size());
    for(int i=0;i<matrix_.size();i++){
      y[i] = matrix_[i][n];
    }
    return y;
  }
  Matrix operator *(const Matrix& l) {
    // 行列のサイズ取得
    std::size_t cols1 = matrix_.size();
    std::size_t rows1 = matrix_[0].size();
    if(cols1 == 1 && rows1 == 1) return matrix_[0][0] * l;
    std::size_t cols2 = l.matrix_.size();
    std::size_t rows2 = l.matrix_[0].size();
    if(cols2 == 1 && rows2 == 1) return operator*(l.matrix_[0][0]);
    std::vector<std::vector<double>> ret_matrix(cols1, std::vector<double>(rows2, 0));
    for(int i = 0; i < cols1; i++){
      for(int s = 0; s < rows2; s++){
        ret_matrix[i][s] = dot(matrix_[i], l.column(s));
      }
    }
    return ret_matrix;
  }
  Matrix operator +(const Matrix& l){
    std::vector<std::vector<double>> ret_matrix;
    for(int r = 0; r < matrix_.size(); r++){
      std::vector<double> vector;
      for(int c = 0; c < matrix_[0].size(); c++){
        vector.push_back(matrix_[r][c] + l.matrix_[r][c]);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }
  Matrix operator -(const Matrix& l) {
    std::vector<std::vector<double>> ret_matrix;
    for (int r = 0; r < matrix_.size(); r++) {
      std::vector<double> vector;
      for (int c = 0; c < matrix_[0].size(); c++) {
        vector.push_back(matrix_[r][c] - l.matrix_[r][c]);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }

  Matrix operator &(const Matrix& l) {
    std::size_t cols1 = matrix_.size();
    std::size_t rows1 = matrix_[0].size();
    std::vector<std::vector<double>> ret_matrix;
    for (int i = 0; i < matrix_.size(); i++) {
      std::vector<double> vector;
      for (int j = 0; j < matrix_[0].size(); j++) {
        vector.push_back(l.matrix_[i][j] * matrix_[i][j]);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }
  Matrix operator *(double scaler) {
    std::vector<std::vector<double>> ret_matrix;
    for (int i = 0; i < matrix_.size(); i++) {
      std::vector<double> vector;
      for (int j = 0; j < matrix_[0].size(); j++) {
        vector.push_back(scaler * matrix_[i][j]);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }
  friend Matrix operator* (double scaler, Matrix m) {
    return m * scaler;
  }

  Matrix t(){
    // 行列のサイズ取得
    std::size_t rows = matrix_.size();
    std::size_t cols = matrix_[0].size();

    // 転置行列の初期化
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows, 0));

    // 転置行列の作成
    for (std::size_t i = 0; i < cols; ++i) {
//      std::vector<double> row;
      for (std::size_t j = 0; j < rows; ++j) {
//        row.push_back(matrix_[j][i]);
        transposed[i][j] = matrix_[j][i];
      }
//      transposed.push_back(row);
    }
    return transposed;
  }

  // 行列を表示するメソッド
  void printMatrix() const {
    std::cout << "------" << std::endl;
    for (const auto& row : matrix_) {
      for (const double & value : row) {
        std::cout << value << " ";
      }
      std::cout << std::endl;
    }
  }
  // 行列の各行にアクセスするための operator[]
  std::vector<double>& operator[](std::size_t index) {
    return matrix_[index];
  }

  // 行列の各行にアクセスするための const operator[]
  const std::vector<double>& operator[](std::size_t index) const {
    return matrix_[index];
  }

  // operator() をオーバーロードして全ての要素に関数を適用
  Matrix operator()(const std::function<double(double &)>& func) {
    // 行列のサイズ取得
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
//    Matrix ret_matrix(cols, rows);
    std::vector<std::vector<double>> ret_matrix(cols, std::vector<double>(rows, 0));
    for (std::size_t i = 0; i < cols; ++i) {
      for (std::size_t j = 0; j < rows; ++j) {
        ret_matrix[i][j] = func(matrix_[i][j]);
      }
    }
    return ret_matrix;
  }
  void push_back(std::vector<double> col){
    matrix_.push_back(col);
  }
  int cols(){
    return matrix_.size();
  }
  int rows(){
    return matrix_[0].size();
  }
};
#endif // STUDY_NN_MATRIX_HPP
