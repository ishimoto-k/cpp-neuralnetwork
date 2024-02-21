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

class MatrixSizeMismatchException : public std::exception {
  std::string msg_;
public:
  MatrixSizeMismatchException(std::string msg){
    msg_ = msg;
  }
  std::string what() {
    std::string msg = "Matrix size mismatch!" + msg_;
    return msg;
  }
};


class Matrix{
  std::vector<std::vector<double>> matrix_ = {};
public:

  static Matrix Flatten(Matrix input){
    std::vector<double> output;
    for(auto col :input.matrix_){
      for(auto d: col){
        output.push_back(d);
      }
    }
    return std::vector<std::vector<double>>{output};
  }
  static Matrix Random(std::size_t size){
    return Random(size, size);
  }
  static Matrix Random(std::size_t cols, std::size_t rows){
    // 乱数エンジンの生成
    std::random_device rd;
    std::mt19937 gen(rd());

    // 範囲指定（0.0から1.0までの実数）
    std::uniform_real_distribution<double> dist(-1, 1);
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
  Matrix(std::size_t cols, std::size_t rows, std::vector<double> data) : matrix_(std::vector<std::vector<double>>(cols, std::vector<double>(rows, 0))) {
    for(int i = 0; i < cols; i++){
      for(int j = 0; j < rows; j++){
        matrix_[i][j] = data[i * rows + j];
      }
    }
  }
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

  Matrix operator +(const double& l){
    std::vector<std::vector<double>> ret_matrix;
    for(int r = 0; r < matrix_.size(); r++){
      std::vector<double> vector;
      for(int c = 0; c < matrix_[0].size(); c++){
        vector.push_back(matrix_[r][c] + l);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }
  Matrix operator -(const double& l) {
    return *this + (-l);
  }

  Matrix operator +(Matrix l){
    if(l.cols() == 1 && l.rows() == 1){
      return *this + l[0][0];
    }
    if(l.cols() != cols()){
      throw MatrixSizeMismatchException("");
    }
    if(l.rows() != rows()){
      throw MatrixSizeMismatchException("");
    }
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
  Matrix operator -(Matrix l) {
    if(l.cols() == 1 && l.rows() == 1){
      return *this - l[0][0];
    }
    if(l.cols() != cols()){
      throw MatrixSizeMismatchException(std::to_string(l.cols()) + "," + std::to_string(cols()));
    }
    if(l.rows() != rows()){
      throw MatrixSizeMismatchException(std::to_string(l.rows()) + "," + std::to_string(rows()));
    }
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
      for (std::size_t j = 0; j < rows; ++j) {
        transposed[i][j] = matrix_[j][i];
      }
    }
    return transposed;
  }

  // 行列を表示するメソッド
  void printMatrix() const {
    std::cout << "------" << std::endl;
    std::cout << "cols:" << cols() << ", rows:" << rows() << std::endl;
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
  int cols() const {
    return matrix_.size();
  }
  int rows() const {
    return matrix_[0].size();
  }
  Matrix padding(int padding = 1){
    // 行列のサイズ取得
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
    std::vector<std::vector<double>> ret_matrix = matrix_;
    for (std::size_t i = 0; i < cols; ++i) {
      for (int j = 0; j < padding; j++){
        ret_matrix[i].insert(ret_matrix[i].begin(), 0);
        ret_matrix[i].insert(ret_matrix[i].end(), 0);
      }
    }
    for (int j = 0; j < padding; j++){
      ret_matrix.insert(ret_matrix.begin(), std::vector<double>(rows+padding*2, 0));
      ret_matrix.insert(ret_matrix.end(), std::vector<double>(rows+padding*2, 0));
    }
    return ret_matrix;
  }
  Matrix convolution(Matrix filter, int stride){
    int cols = matrix_.size();
    int rows = matrix_[0].size();
    int filter_size = filter.cols();
    int filter_radius = filter_size / 2; // フィルターの中心からの距離

    // 出力画像のサイズの計算
    int output_height = (cols - filter_size) / stride + 1;
    int output_width = (rows - filter_size) / stride + 1;

    // 出力画像の初期化

    std::vector<std::vector<double>> ret_matrix(output_height, std::vector<double>(output_width, 0));

    // 畳み込み演算
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        float sum = 0.0f;
        for (int i = 0; i < filter_size; ++i) {
          for (int j = 0; j < filter_size; ++j) {
            int image_y = y * stride + i;
            int image_x = x * stride + j;
            sum += matrix_[image_y][image_x] * filter[i][j];
          }
        }
        ret_matrix[y][x] = sum;
      }
    }
    return ret_matrix;
  }

  double average(){
    double n = cols() * rows();
    double ans = 0;
    for(int y = 0; y < cols(); y++){
      for(int x = 0; x < rows(); x++){
        ans += matrix_[y][x]/n;
      }
    }
    return ans;
  }

  // 畳み込み層の誤差逆伝播を行う関数
  static Matrix backward_convolution(Matrix input, Matrix output_error, Matrix filter, int stride) {
    int input_height = input.cols();
    int input_width = input.rows();
    int filter_height = filter.cols();
    int filter_width = filter.rows();
    int output_height = output_error.cols();
    int output_width = output_error.rows();

    Matrix filter_error = Matrix(filter_height, filter_width);

    // フィルタの重みの勾配を計算
    for (int i = 0; i < filter_height; ++i) {
      for (int j = 0; j < filter_width; ++j) {
        for (int k = 0; k < output_height; ++k) {
          for (int l = 0; l < output_width; ++l) {
            filter_error[i][j] += input[k * stride + i][l * stride + j] * output_error[k][l];
          }
        }
      }
    }
    return filter_error;
  }


  static Matrix pooling(Matrix input, int pool_size, int stride){
    int cols = input.cols();
    int rows = input.rows();
    int output_height = (cols - pool_size) / stride + 1;
    int output_width = (rows - pool_size) / stride + 1;
    std::vector<std::vector<double>> pooled(output_height, std::vector<double>(output_width, 0));
    for (int i = 0; i < output_height; ++i) {
      for (int j = 0; j < output_width; ++j) {
        double max_val = std::numeric_limits<double>::min();
        for (int m = 0; m < pool_size; ++m) {
          for (int n = 0; n < pool_size; ++n) {
            max_val = std::max(max_val, input[i * stride + m][j * stride + n]);
          }
        }
        pooled[i][j] = max_val;
      }
    }
    return pooled;
  }

  static Matrix grad_pooling(Matrix input, Matrix grad_output, int pool_size, int stride) {
    int cols = input.cols();
    int rows = input.rows();
    int grad_height = grad_output.cols();
    int grad_width = grad_output.rows();

    std::vector<std::vector<double>> grad_input(cols, std::vector<double>(rows, 0));

    for (int i = 0; i < grad_height; ++i) {
      for (int j = 0; j < grad_width; ++j) {
        float max_val = input[i * stride][j * stride];
        int max_row = i * stride;
        int max_col = j * stride;
        for (int m = 0; m < pool_size; ++m) {
          for (int n = 0; n < pool_size; ++n) {
            if (input[i * stride + m][j * stride + n] > max_val) {
              max_val = input[i * stride + m][j * stride + n];
              max_row = i * stride + m;
              max_col = j * stride + n;
            }
          }
        }
        grad_input[max_row][max_col] = grad_output[i][j];
      }
    }
    return grad_input;
  }

};

#endif // STUDY_NN_MATRIX_HPP
