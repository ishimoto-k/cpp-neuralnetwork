//
// Created by IshimotoKiko on 2023/12/28.
//

#include "matrix/Matrix.hpp"
#include <fstream>
#include <regex>
#include <sstream>
#include <AffineLayer.hpp>
#include <SigmoidLayer.hpp>
#include <BatchNormalizeLayer.hpp>

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_csv(const std::string& file_name) {
  std::vector<std::vector<double>> data;
  std::vector<std::vector<double>> teacher_data;

  std::ifstream file(file_name);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << file_name << std::endl;
    exit(0);
  }

  int t = 0;
  int count = 0;
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
    count ++;
  }

  return {data, teacher_data};
}

int main() {
#define MIDDLE_NUM 128
#define OUTPUT_NUM 10
#define BATCH_SIZE 5
  BatchNormalizeLayer batchNormalizeLayer(3, 2, 0.1);
  Matrix matrix_in = std::vector<std::vector<double>>{{0,1,2}, {0,0.1,0.2}};
  Matrix matrix_out = std::vector<std::vector<double>>{{0,0.1,0.1}, {-0.1,1,0.2}};
  auto ret = batchNormalizeLayer.forward(matrix_in);
  ret.printMatrix();
  ret = batchNormalizeLayer.backward(matrix_out);
  ret.printMatrix();

  auto r = read_csv(MNIST_TRAIN_PATH);
  const double learning_rate = 0.1;

  std::shared_ptr<BaseLayer> affine_layer_0 = std::make_shared<AffineLayer>(r.first[0].size(), MIDDLE_NUM, learning_rate);
  std::shared_ptr<BaseLayer> sigmoid_layer_0 = std::make_shared<SigmoidLayer>(MIDDLE_NUM);
  std::shared_ptr<BaseLayer> affine_layer_1 = std::make_shared<AffineLayer>(MIDDLE_NUM, OUTPUT_NUM, learning_rate);
  std::shared_ptr<BaseLayer> sigmoid_layer_1 = std::make_shared<SigmoidLayer>(OUTPUT_NUM);
  //training
  for(int epoch = 0; epoch < r.first.size(); epoch++){
    std::cout << epoch << "/" << r.first.size() << std::endl;
    auto teacher_input = std::vector<std::vector<double>>{};
    auto teacher_output = std::vector<std::vector<double>>{};
    for(int batch = 0; batch < BATCH_SIZE; batch++){
      teacher_input.push_back(r.first[epoch]);
      teacher_output.push_back(r.second[epoch]);
      epoch++;
    }
    Matrix inputs = teacher_input;
    Matrix teacher = teacher_output;

    std::cout << "teacher"; teacher.printMatrix();
    //      std::cout << "inputs"; inputs.printMatrix();

    //forward
    auto affine_0 = affine_layer_0->forward(inputs);
    auto sigmoid_0 = sigmoid_layer_0->forward(affine_0);
    auto affine_1 = affine_layer_1->forward(sigmoid_0);
    auto output = sigmoid_layer_1->forward(affine_1);
    std::cout << "output"; output.printMatrix();


    //backward
    auto output_error = (output - teacher);
    auto sigmoid_grad1 = sigmoid_layer_1->backward(output_error);
    auto affine_grad1 = affine_layer_1->backward(sigmoid_grad1);
    auto sigmoid_grad0 = sigmoid_layer_0->backward(affine_grad1);
    auto finish = affine_layer_0->backward(sigmoid_grad0);

    sigmoid_layer_1->update();
    affine_layer_1->update();
    sigmoid_layer_0->update();
    affine_layer_0->update();

  }




  auto test = read_csv("/Users/ishimotokiko/Desktop/program/mnist_test.csv");
  for (int i=0; i<test.first.size(); i++){
    std::cout << i << "/" << test.first.size() << std::endl;

    Matrix inputs = std::vector<std::vector<double>> {test.first[i]};
    Matrix answer = std::vector<std::vector<double>> {test.second[i]};

    //forward
    auto affine_0 = affine_layer_0->forward(inputs);
    auto sigmoid_0 = sigmoid_layer_0->forward(affine_0);
    auto affine_1 = affine_layer_1->forward(sigmoid_0);
    auto output = sigmoid_layer_1->forward(affine_1);
    std::cout << "answer"; answer.printMatrix();
    std::cout << "output"; output.printMatrix();

  }
}