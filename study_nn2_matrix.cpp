#include <iostream>
#include <utility>
#include <vector>
#include <functional>
#include <cmath>

class Matrix{
  std::vector<std::vector<double>> matrix_;
public:
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

  Matrix(int cols, int rows){
    std::vector<std::vector<double>> initialize(cols, std::vector<double>(rows, 0));
    matrix_ = initialize;
  }
  Matrix(std::vector<std::vector<double>> init){
    matrix_ = std::move(init);
  }
  // 代入演算子のオーバーロード
  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      // 自己代入を防ぐために確認
      matrix_ = other.matrix_;
    }
    return *this;
  }

  // コピーコンストラクタ
  Matrix(const Matrix& other) : matrix_(other.matrix_) {}

  Matrix operator *(const Matrix& l) {
    // 行列のサイズ取得
    std::size_t cols1 = matrix_.size();
    std::size_t rows1 = matrix_[0].size();
    std::size_t cols2 = l.matrix_.size();
    std::size_t rows2 = l.matrix_[0].size();
    std::vector<std::vector<double>> ret_matrix(cols1, std::vector<double>(rows2, 0));
//    Matrix ret_matrix(cols1, rows2);
    for (int y = 0; y < ret_matrix.size(); y++){
      for(int x = 0; x < ret_matrix[y].size(); x++){
        for(int r = 0; r < rows1; r++) {
          ret_matrix[x][y] += matrix_[y][r] * l.matrix_[r][x];
        }
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
        transposed[j][i] = matrix_[i][j];
      }
    }
    return transposed;
  }

  // 行列を表示するメソッド
  void printMatrix() const {
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
    std::vector<std::vector<double>> ret_matrix(cols, std::vector<double>(rows, 0));

    for (std::size_t i = 0; i < cols; ++i) {
      for (std::size_t j = 0; j < rows; ++j) {
        ret_matrix[i][j] = func(matrix_[i][j]);
      }
    }
    return ret_matrix;
  }
};

// シグモイド関数
double sigmoid(double x) {
  return 1 / (1 + std::exp(-x));
}

int main() {
  Matrix matrix({{1, 2, 3}, {4, 5, 6}});
  matrix.printMatrix();
  //1 2 3
  //4 5 6

  {
    std::cout << "addition" << std::endl;
    Matrix matrix_left({{2, 3, 4}, {14, 15, 16}});

    auto ans = matrix + matrix_left;
    ans.printMatrix();
    //addition
    //3 5 7
    //18 20 22
  }
  {
    std::cout << "subtraction" << std::endl;
    Matrix matrix_left({{2, 3, 4}, {14, 15, 16}});

    auto ans = matrix - matrix_left;
    ans.printMatrix();
    //subtraction
    //-1 -1 -1
    //-10 -10 -10
  }
  {
    std::cout << "multiplication 1" << std::endl;
    Matrix matrix_r({{2, 3},
                     {1, 1}});
    Matrix matrix_l({{2, 3},
                        {14, 15}});

    auto ans = matrix_l * matrix_r;
    ans.printMatrix();
    //multiplication 1
    // 7 43
    // 9 57
  }
  {
    std::cout << "multiplication 2" << std::endl;
    Matrix matrix_l({{2, 3, 4}});
    Matrix matrix_r({{2},
                     {14},
                     {10}});

    auto ans = matrix_l * matrix_r;
    ans.printMatrix();
    //multiplication 2
    //86
  }
  {
    std::cout << "scalar multiplication" << std::endl;
    double coefficient = 0.5;

    auto ans = coefficient * matrix;
    ans.printMatrix();
    //scalar multiplication
    //0.5 1 1.5
    //2 2.5 3
  }
  {
    std::cout << "per-element function" << std::endl;
    auto ans = matrix(sigmoid);
    ans.printMatrix();
    //per-element function
    //0.731059 0.880797 0.952574
    //0.982014 0.993307 0.997527
  }
}
