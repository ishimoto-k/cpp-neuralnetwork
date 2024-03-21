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

class Point{
public:
  Point(int x, int y): x(x), y(y){

  }
  int x;
  int y;
};
enum VECTOR{
  ROW,
  COL
};
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

  static std::vector<Matrix> FromFlatten(int cols, int rows, std::vector<double> data, int num){
    std::vector<Matrix> ret;
    for(int c=0; c<num; c++){
      Matrix matrix(cols, rows);
      for(int i = 0; i < cols; i++){
        for(int j = 0; j < rows; j++){
          matrix[i][j] = data[c*cols*rows + i * rows + j];
        }
      }
      ret.push_back(matrix);
    }
    return ret;
  }
  static Matrix Flatten(std::vector<Matrix> inputs){
    std::vector<double> output;
    for(auto matrix: inputs){
      for(auto col :matrix.matrix_){
        for(auto d: col){
          output.push_back(d);
        }
      }
    }
    return std::vector<std::vector<double>>{output};
  }
  static Matrix Random(std::size_t size){
    return Random(size, size);
  }
  static Matrix Random(std::size_t cols, std::size_t rows, std::normal_distribution<double> dist){
    // 乱数エンジンの生成
    std::random_device rd;
    std::mt19937 gen(rd());
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

  static Matrix RandomHe(std::size_t cols, std::size_t rows){
    // 範囲指定（0.0から1.0までの実数）
    double he_stddev = sqrt(2.0 / cols);
    std::normal_distribution<double> dist(0.0, he_stddev);
    return Random(cols, rows, dist);
  }
  static Matrix RandomXavier(std::size_t cols, std::size_t rows){
    // 範囲指定（0.0から1.0までの実数）
    double xavier_stddev = sqrt(2.0 / double(cols + rows));
    std::normal_distribution<double> dist(0.0, xavier_stddev);
    return Random(cols, rows, dist);
  }
  static Matrix Random(std::size_t cols, std::size_t rows){
    // 乱数エンジンの生成
    std::random_device rd;
    std::mt19937 gen(rd());

    // 範囲指定（0.0から1.0までの実数）
    std::uniform_real_distribution<double> dist(-1, 1);
    //    std::normal_distribution<double> dist(0.0, 0.01);
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
  static Matrix RandomNormal(std::size_t cols, std::size_t rows){
    std::normal_distribution<double> dist(0.0, 0.1);
    return Random(cols, rows, dist);
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
  Matrix() : matrix_(std::vector<std::vector<double>>(0, std::vector<double>(0, 0))){

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
  Matrix operator -() {
    Matrix ret = matrix_;
    return (*this)([](double x){
      return -x;
    });
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
      if(l.rows() == rows() && l.cols() == 1){
        std::vector<std::vector<double>> ret_matrix;
        for(int r = 0; r < matrix_.size(); r++){
          std::vector<double> vector;
          for(int c = 0; c < matrix_[0].size(); c++){
            vector.push_back(matrix_[r][c] + l.matrix_[0][c]);
          }
          ret_matrix.push_back(vector);
        }
        return ret_matrix;
      }
      throw MatrixSizeMismatchException("");
    }
    if(l.rows() != rows()){
      if(l.cols() == cols() && l.rows() == 1){
        std::vector<std::vector<double>> ret_matrix;
        for(int r = 0; r < matrix_.size(); r++){
          std::vector<double> vector;
          for(int c = 0; c < matrix_[0].size(); c++){
            vector.push_back(matrix_[r][c] + l.matrix_[r][0]);
          }
          ret_matrix.push_back(vector);
        }
        return ret_matrix;
      }
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
    return *this + (-l);
  }

  Matrix operator /(const Matrix& l) {
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
    std::size_t l_cols = l.matrix_.size();
    std::size_t l_rows = l.matrix_[0].size();
    std::vector<std::vector<double>> ret_matrix;
    for (int i = 0; i < cols; i++) {
      std::vector<double> vector;
      for (int j = 0; j < rows; j++) {
        vector.push_back(matrix_[i][j] / l.matrix_[i%l_cols][j%l_rows]);
      }
      ret_matrix.push_back(vector);
    }
    return ret_matrix;
  }
  Matrix operator &(const Matrix& l) {
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
    std::size_t l_cols = l.matrix_.size();
    std::size_t l_rows = l.matrix_[0].size();
    std::vector<std::vector<double>> ret_matrix;
    for (int i = 0; i < cols; i++) {
      std::vector<double> vector;
      for (int j = 0; j < rows; j++) {
        vector.push_back(matrix_[i][j] * l.matrix_[i%l_cols][j%l_rows]);
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

  Matrix sum(VECTOR vector){
    if(vector == VECTOR::ROW){
      Matrix ret(cols(), 1);
      for(int col = 0; col < cols(); col++){
        for(int i = 0; i < rows(); i++) {
          ret[col][0] += matrix_[col][i];
        }
      }
      return ret;
    }
    else if(vector == VECTOR::COL){
      Matrix ret(1, rows());
      for(int row = 0; row < rows(); row++){
        for(int i = 0; i < cols(); i++) {
          ret[0][row] += matrix_[i][row];
        }
      }
      return ret;
    }
  }
  Matrix mean(VECTOR vector){
    if(vector == VECTOR::ROW){
      Matrix ret(cols(), 1);
      for(int col = 0; col < cols(); col++){
        for(int i = 0; i < rows(); i++) {
          ret[col][0] += matrix_[col][i]/matrix_[col].size();
        }
      }
      return ret;
    }
    else if(vector == VECTOR::COL){
      Matrix ret(1, rows());
      for(int row = 0; row < rows(); row++){
        for(int i = 0; i < cols(); i++) {
          ret[0][row] += matrix_[i][row]/matrix_.size();
        }
      }
      return ret;
    }
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
  void printSize() const {
    std::cout << "------" << std::endl;
    std::cout << "cols:" << cols() << ", rows:" << rows() << std::endl;
  }
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

  void push_back(Matrix matrix){
    if(matrix.cols() == 1){
      if(rows() == matrix.rows()) {
        matrix_.push_back(matrix[0]);
      }else{
        throw MatrixSizeMismatchException("");
      }
    }
    else if(matrix.rows() == 1){
      if(cols() == matrix.cols()) {
        for(int i = 0; i<cols(); i++){
          matrix_[i].push_back(matrix[i][0]);
        }
      }else{
        throw MatrixSizeMismatchException("");
      }
    }
  }
  Matrix col(int index){
    auto ret = std::vector<std::vector<double>>(1, std::vector<double>(rows(), 0));
    for(int row = 0; row < rows(); row++){
      ret[0][row] = matrix_[index][row];
    }
    return ret;
  }
  Matrix row(int index){
    auto ret = std::vector<std::vector<double>>(cols(), std::vector<double>(1, 0));
    for(int col = 0; col < cols(); col++){
      ret[col][0] = matrix_[col][index];
    }
    return ret;
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
  Matrix operator()(const std::function<double(double &, Matrix*, int, int)>& func, Matrix* matrix) {
    // 行列のサイズ取得
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
    //    Matrix ret_matrix(cols, rows);
    std::vector<std::vector<double>> ret_matrix(cols, std::vector<double>(rows, 0));
    for (std::size_t i = 0; i < cols; ++i) {
      for (std::size_t j = 0; j < rows; ++j) {
        ret_matrix[i][j] = func(matrix_[i][j], matrix, i, j);
      }
    }
    return ret_matrix;
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
  // 180度回転
  Matrix rotate180() {
    std::size_t cols = matrix_.size();
    std::size_t rows = matrix_[0].size();
    std::vector<std::vector<double>> rotated(cols, std::vector<double>(rows));
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        rotated[cols - i - 1][rows - j - 1] = matrix_[i][j];
      }
    }
    return rotated;
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
  Matrix convolution(Matrix& filter, int stride){
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

  double sum(){
    double n = cols() * rows();
    double ans = 0;
    for(int y = 0; y < cols(); y++){
      for(int x = 0; x < rows(); x++){
        ans += matrix_[y][x];
      }
    }
    return ans;
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

  static Matrix deconvolution(Matrix output, Matrix kernel, int stride) {
    int outputHeight = output.cols();
    int outputWidth  = output.rows();
    int kernelHeight = kernel.cols();
    int kernelWidth  = kernel.rows();
    int inputHeight  = (outputHeight - 1) * stride + kernelHeight;
    int inputWidth   = (outputWidth - 1) * stride + kernelWidth;

    std::vector<std::vector<double>> input(inputHeight, std::vector<double>(inputWidth, 0));

    for (int i = 0; i < outputHeight; ++i) {
      for (int j = 0; j < outputWidth; ++j) {
        for (int k = 0; k < kernelHeight; ++k) {
          for (int l = 0; l < kernelWidth; ++l) {
            input[i * stride + k][j * stride + l] += output[i][j] * kernel[k][l];
          }
        }
      }
    }

    return input;
  }
  static Matrix updateFilter(Matrix input, Matrix outputGradient, int stride) {
    int inputHeight  = input.cols(); // 入力行列の高さ
    int inputWidth   = input.rows(); // 入力行列の幅
    int outputHeight = outputGradient.cols(); // 出力勾配行列の高さ
    int outputWidth  = outputGradient.rows(); // 出力勾配行列の幅
    int kernelHeight = outputHeight - stride + 1; // カーネルの高さ
    int kernelWidth  = outputWidth - stride + 1; // カーネルの幅

    std::vector<std::vector<double>> filterGradient(kernelHeight, std::vector<double>(kernelWidth, 0)); // フィルタの勾配行列を初期化
    // フィルタの勾配を計算
    for (int i = 0; i < kernelHeight; ++i) {
      for (int j = 0; j < kernelWidth; ++j) {
        for (int k = 0; k < outputHeight; ++k) {
          for (int l = 0; l < outputWidth; ++l) {
            if (i + k < inputHeight && j + l < inputWidth) { // 範囲内の場合にのみ計算する
              filterGradient[i][j] += input[i + k][j + l] * outputGradient[k][l]; // フィルタの勾配を計算
            }
          }
        }
      }
    }

    return filterGradient; // 計算されたフィルタの勾配行列を返す
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
        double max_val = std::numeric_limits<double>::lowest();
        int max_x = 0, max_y = 0;
        for (int m = 0; m < pool_size; ++m) {
          for (int n = 0; n < pool_size; ++n) {
            double in = input[i * stride + m][j * stride + n];
            if(in > max_val){
              max_y = i * stride + m;
              max_x = j * stride + n;
              max_val = in;
            }
          }
        }
        pooled[i][j] = max_val;
      }
    }
    return pooled;
  }

  static Matrix grad_pooling(Matrix grad, Matrix input, int pool_size, int stride) {
    int cols = input.cols();
    int rows = input.rows();
    int output_height = (cols - pool_size) / stride + 1;
    int output_width = (rows - pool_size) / stride + 1;
    std::vector<std::vector<double>> ret(cols, std::vector<double>(rows, 0));
    for (int i = 0; i < output_height; ++i) {
      for (int j = 0; j < output_width; ++j) {
        double max_val = std::numeric_limits<double>::lowest();
        int max_x = 0, max_y = 0;
        for (int m = 0; m < pool_size; ++m) {
          for (int n = 0; n < pool_size; ++n) {
            double in = input[i * stride + m][j * stride + n];
            if(in > max_val){
              max_y = i * stride + m;
              max_x = j * stride + n;
              max_val = in;
            }
          }
        }
        ret[max_y][max_x] = grad[i][j];
      }
    }
    return ret;
  }


  // Softmax関数
  static Matrix softmax(const Matrix& x) {
    std::vector<double> y;
    double sum_exp = 0.0;
    for (double val : x[0]) {
      sum_exp += std::exp(val);
    }
    for (double val : x[0]) {
      y.push_back(std::exp(val) / sum_exp);
    }
    return std::vector<std::vector<double>>{y};
  }

  // Cross Entropy Error
  static double cross_entropy_error(const Matrix& y, const Matrix& t) {
    if (y[0].size() != t[0].size()) {
      std::cerr << "Error: ラベルと予測値の次元数が異なります。" << std::endl;
      return 0.0;
    }

    double error = 0.0;
    for (size_t i = 0; i < y[0].size(); ++i) {
      error += -t[0][i] * std::log(y[0][i] + 1e-7); // 0除算を防ぐために微小な値を加算
    }

    return error;
  }

  // SoftmaxWithLossの forward
  static double SoftmaxWithLossForward(const Matrix& x, const Matrix& t) {
    auto y = softmax(x);
    double loss = cross_entropy_error(y, t);
    return loss;
  }

  // SoftmaxWithLossの backward
  static Matrix SoftmaxWithLossBackward(const Matrix& x, const Matrix& t) {
    auto y = softmax(x);
    std::vector<double> grad;
    for (size_t i = 0; i < y[0].size(); ++i) {
      grad.push_back((y[0][i] - t[0][i]) / static_cast<double>(x[0].size()));
    }
    return std::vector<std::vector<double>>{grad};
  }

  // 順伝播
  static Matrix softmax_forward(Matrix input) {
    // 入力値から最大値を引く（オーバーフロー対策）
    double max_val = *std::max_element(input[0].begin(), input[0].end());
    std::vector<double> input_shifted;
    for (double val : input[0]) {
      input_shifted.push_back(val - max_val);
    }

    // softmax計算
    std::vector<std::vector<double>> output;
    std::vector<double> tmp;
    double sum_exp = 0.0;
    for (double val : input_shifted) {
      double exp_val = std::exp(val);
      sum_exp += exp_val;
      tmp.push_back(exp_val);
    }

    // 合計値で割って正規化
    for (double& val : tmp) {
      val /= sum_exp;
    }
    return std::vector<std::vector<double>>{tmp};
  }

  // 逆伝播
  // Softmax関数の逆伝播
  static Matrix softmax_backward(const Matrix& teacher, const Matrix& output) {
    int batch_size = teacher.cols() * teacher.rows();
    Matrix dx(output.cols(), output.rows());
    // 正解ラベルと出力のサイズが等しい場合
    for (int i = 0; i < output.cols(); ++i) {
      for (int j = 0; j < output.rows(); ++j) {
        dx[i][j] = (output[i][j] - teacher[i][j]);
      }
    }
    return dx;
  }
  static Matrix im2col(const Matrix& input, int filter_height, int filter_width, int stride) {
    std::vector<std::vector<double>> result;

    int input_height = input.cols();
    int input_width = input.rows();

    // 畳み込み演算の適用回数を計算
    int output_height = (input_height - filter_height) / stride + 1;
    int output_width = (input_width - filter_width) / stride + 1;

    // im2col処理
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        std::vector<double> col;
        for (int fy = 0; fy < filter_height; ++fy) {
          for (int fx = 0; fx < filter_width; ++fx) {
            int input_y = y * stride + fy;
            int input_x = x * stride + fx;
            col.push_back(input[input_y][input_x]);
          }
        }
        result.push_back(col);
      }
    }

    return result;
  }
  static Matrix col2im(const Matrix& col, int input_rows, int input_cols, int kernel_size, int stride) {
    std::vector<std::vector<double>> im_data(input_rows, std::vector<double>(input_cols, 0));

    for (int i = 0; i < col.rows(); ++i) {
      int col_index = i;
      int row_offset = (i / (input_cols / kernel_size)) * stride;
      int col_offset = (i % (input_cols / kernel_size)) * stride;
      for (int m = 0; m < kernel_size; ++m) {
        for (int n = 0; n < kernel_size; ++n) {
          im_data[row_offset + m][col_offset + n] += col[col_index][m * kernel_size + n];
        }
      }
    }
    return im_data;
  }

  // 勾配クリッピングの関数
  static Matrix clip_gradients(Matrix& gradients, double threshold) {
    std::vector<std::vector<double>> im_data(gradients.cols(), std::vector<double>(gradients.rows(), 0));
    for (int i=0; i<gradients.cols(); i++){
      for (int j=0; j<gradients.rows(); j++){
        double val = gradients[i][j];
        if (val > threshold) {
          std::cout << "overflow: " << val << std::endl;
          val = threshold;
        } else if (val < -threshold) {
          std::cout << "overflow: " << val << std::endl;
          val = -threshold;
        }
        gradients[i][j] = val;
        im_data[i][j] = val;
      }
    }
    return im_data;
  }
};

#endif // STUDY_NN_MATRIX_HPP
