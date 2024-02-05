//
// Created by IshimotoKiko on 2023/12/28.
//

#include "matrix/Matrix.hpp"
#include <fstream>
#include <regex>
#include <sstream>

double sigmoid_(double x){
  return 1.0 / (1.0 + std::exp(-x));
}
double sigmoid_derivative(double x){
  return x * (1.0 - x);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_csv(const std::string& file_name) {
  std::vector<std::vector<double>> data;
  std::vector<std::vector<double>> teacher_data;

  std::ifstream file(file_name);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << file_name << std::endl;
    exit(0);
  }

  int t = 0;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;
    int i = 0;
    std::vector<double> tmp;
    while (std::getline(ss, token, ',')) {
      try {
        if(i==0){
          t = std::stoi(token);
        }else {
          double value = std::stod(token)/255;
          tmp.push_back(value);
        }
      } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
      } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
      }
      i++;
    }

    std::vector<double> teacher;
    for(int i=0; i<10; i++){
      if(i==t) {
        teacher.push_back(1);
      }else {
        teacher.push_back(0);
      }
    }
    teacher_data.push_back(teacher);
    data.push_back(tmp);
  }

  return {data, teacher_data};
}


int main() {
#define INPUT_NUM 2
#define MIDDLE_NUM 128
#define OUTPUT_NUM 10
  auto r = read_csv("mnist_train.csv");
  const double learning_rate = 0.1;
  Matrix middle_weights = Matrix::Random(MIDDLE_NUM, r.first[0].size());
  Matrix middle_bias_weights = Matrix::Random(MIDDLE_NUM, 1);
  Matrix output_weights = Matrix::Random(OUTPUT_NUM, MIDDLE_NUM);
  Matrix output_bias_weights = Matrix::Random(OUTPUT_NUM, 1);
//  std::cout << "middle_weights"; middle_weights.printMatrix();
//  std::cout << "output_weights"; output_weights.printMatrix();
  std::function<double(double)> activation_function = sigmoid_;
  std::function<double(double)> activation_function_derivative = sigmoid_derivative;
  //training
  for(int epoch = 0; epoch < r.first.size(); epoch++){
    std::cout << epoch << "/" << r.first.size() << std::endl;
    Matrix teacher_input = std::vector<std::vector<double>>{r.first[epoch]};
    Matrix teacher_output = std::vector<std::vector<double>>{r.second[epoch]};

    for(int pattern_i = 0; pattern_i < teacher_input.cols(); pattern_i++){
      Matrix inputs = std::vector<std::vector<double>> {teacher_input[pattern_i]};
      Matrix teacher = Matrix(std::vector<std::vector<double>>{teacher_output[pattern_i]});
      std::cout << "teacher"; teacher.printMatrix();
//      std::cout << "inputs"; inputs.printMatrix();

      //forward
//      std::cout << "forward" << std::endl;
      auto middle_output = (middle_weights * inputs.t() + middle_bias_weights)(sigmoid_);
      auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
      std::cout << "output"; output.t().printMatrix();


      //backward
//      std::cout << "backward" << std::endl;
      auto output_error = (output - teacher.t()) & output(sigmoid_derivative);
//      std::cout << "output_error"; output_error.printMatrix();
//      std::cout << "middle_output"; middle_output.printMatrix();
      auto middle_error = (output_error.t() * output_weights);
      auto bias_output_delta = learning_rate * output_error;
      output_weights = output_weights - bias_output_delta * middle_output.t();
      output_bias_weights = output_bias_weights - bias_output_delta;
//      std::cout << "output_weights"; output_weights.printMatrix();


//      std::cout << std::endl << "update middle_weights"  << std::endl;
      auto bias_middle_delta = learning_rate * (middle_error.t() & middle_output(sigmoid_derivative));
      middle_weights = middle_weights - bias_middle_delta * inputs;
      middle_bias_weights = middle_bias_weights - bias_middle_delta;
//      std::cout << "middle_weights"; middle_weights.printMatrix();
//      std::cout << "end" << std::endl << std::endl;
    }
  }




  auto test = read_csv("mnist_test.csv");
  for (int i=0; i<test.first.size(); i++){
    std::cout << i << "/" << test.first.size() << std::endl;

    Matrix inputs = std::vector<std::vector<double>> {test.first[i]};
    Matrix answer = std::vector<std::vector<double>> {test.second[i]};

    //forward
    auto middle_output = (middle_weights * inputs.t() + middle_bias_weights)(sigmoid_);
    auto output = (output_weights * middle_output + output_bias_weights)(sigmoid_);
    std::cout << "answer"; answer.printMatrix();
    std::cout << "output"; output.t().printMatrix();

  }
}
