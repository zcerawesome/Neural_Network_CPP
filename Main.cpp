#include <iostream>
#include "matrice.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Network.h"
#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>

bool isNan(matrice<float>& inp)
{
    for(auto& rows: inp.matrix)
        for(auto& col: rows)
            if(std::isnan(col))
                return true;
    return false;
}

std::vector<std::vector<float>> loadCSV(std::string fileName, bool header=true)
{
    std::vector<std::vector<float>> data;
    std::ifstream file(fileName);
    std::string line;
    if(header)
        getline(file, line);
    int i = 0;
    while(getline(file, line) && i < 1000)
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while(std::getline(ss, value, ','))
        {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
        i++;
    }
    return data;
}

float randomFloat()
{
    return (float)rand() / RAND_MAX;
}

void randomize_matrix(matrice<float>& inp)
{
    for(auto& rows: inp.matrix)
        for(auto& col: rows)
            col = randomFloat() - 0.5;
}

matrice<float> ReLU(matrice<float>& inp)
{
    matrice<float> temp(inp.getNumRows(), inp.getNumCols());
    for(int i = 0; i < temp.getNumRows(); i++)
        for(int j = 0; j < temp.getNumCols(); j++)
            temp[i][j] = inp[i][j] > 0? inp[i][j]: 0;
    return temp;
}

matrice<float> ReLU_derive(matrice<float>& inp)
{
    matrice<float> temp(inp.getNumRows(), inp.getNumCols());
    for(int i = 0; i < temp.getNumRows(); i++)
        for(int j = 0; j < temp.getNumCols(); j++)
            temp[i][j] = inp[i][j] > 0;
    return temp;
}

matrice<float> softmax(matrice<float>& inp)
{
    matrice<float> temp(inp.getNumRows(), inp.getNumCols());
    for(int j = 0; j < temp.getNumCols(); j++)
    {
        float sum = 0;
        for(int i = 0; i < temp.getNumRows(); i++)
        {
            temp[i][j] = exp(inp[i][j]);
            sum += temp[i][j];
        }
        for(int i = 0; i < temp.getNumRows(); i++)
        {
            temp[i][j] /= sum;
        }
    }
    return temp;
}

vec(matrice<float>) init_params()
{
    matrice<float> W1(10, 784);
    matrice<float> b1(10, 1);
    matrice<float> W2(10, 10);
    matrice<float> b2(10, 1);
    randomize_matrix(W1);
    randomize_matrix(b1);
    randomize_matrix(W2);
    randomize_matrix(b2);
    return {W1, b1, W2, b2};
}

vec(matrice<float>) forward_prop(matrice<float>& W1, matrice<float>& b1, matrice<float>& W2, matrice<float>& b2, matrice<float>& X)
{
    auto Z1 = W1 * X + b1;
    auto A1 = ReLU(Z1);

    auto Z2 = W2 * A1 + b2;
    auto A2 = softmax(Z2);
    return {Z1, A1, Z2, A2};
}

matrice<float> one_hot_encode(matrice<float> Y)
{
    matrice<float> one_hot_encode(10, Y.getNumCols());
    for(int i = 0; i < Y.getNumCols(); i++)
        one_hot_encode[Y[0][i]][i] = 1.0f;
    return one_hot_encode;
}

float sum(matrice<float>& input)
{
    float total = 0;
    for(auto& rows: input.matrix)
        for(auto& cols: rows)
            total += cols;
    return total;
}

vec(matrice<float>) backward_prop(matrice<float>& Z1, matrice<float>& A1, matrice<float>& Z2, matrice<float>& A2, matrice<float>& W1, matrice<float>& W2, matrice<float>& X, matrice<float>& Y)
{
    matrice<float> one_hot_encode_Y = one_hot_encode(Y);
    int cols = Y.getNumCols();
    auto dZ2 = A2 - one_hot_encode_Y;
    auto dW2 = dZ2 * A1.transpose() / cols;
    auto db2 = 1 / cols * sum(dZ2);
    matrice<float> dB2(1,1);
    dB2[0][0] = db2;
    auto dZ1 = (W2.transpose() * dZ2).dot(ReLU_derive(Z1));
    auto dW1 = dZ1 * X.transpose() / cols;
    auto db1 = 1 / cols * sum(dZ1);
    matrice<float> dB1(1,1);
    dB1[0][0] = db1;

    return {dW1, dB1, dW2, dB2};
}

vec(matrice<float>) update_params(matrice<float>& W1, matrice<float>& b1, matrice<float>& W2, matrice<float>& b2, matrice<float>& dW1, matrice<float>& db1, matrice<float>& dW2, matrice<float>& db2, float alpha)
{
    W1 = W1 - dW1 * alpha;
    b1 = b1 - (db1 * alpha)[0][0];
    W2 = W2 - dW2 * alpha;
    b2 = b2 - (db2 * alpha)[0][0];
    return {W1, b1, W2, b2};
}

int largest_index(matrice<float>& input)
{
    int index = 0;
    for(int i = 0; i < input.getNumRows(); i++)
        index = input[i][0] > input[index][0]? i: index;
    return index;
}

int main()
{ 
    Network n;
    n.addLayer(784, 0, 0);
    n.addLayer(10, ReLU, ReLU_derive);
    n.addLayer(10, softmax, 0);

    n.setRandomization(randomize_matrix);
    n.applyRandomzation(1);
    n.applyRandomzation(2);
    
    srand(time(0)+45);
    {
        // vec2D(float) ting({{0,1,2,3}, {0,1,2,3}, {4, 5, 6,7 }, {8, 9, 10, 11}});
        // matrice<float> temp(ting);
        // matrice<float> temp2 = temp.iloc(1,3);
        // matrice<float> temp3 = temp.getRows(1,3);
        // temp.toString();
        // temp2.toString();
        // temp3.toString();

        // vec2D(float) thing({{1,1}});
        // matrice<float> temp3(thing);
        // temp3.toString();
        // temp3.transpose().toString();

    }
    
    std::string fileName = "mnist/train.csv";
    std::vector<std::vector<float>> df = loadCSV(fileName);
    matrice<float> data(df);
    data = data.transpose();
    int rows = data.getNumRows();
    int cols = data.getNumCols();

    matrice<float> test = data.iloc(0, 200);
    matrice<float> X_test = test.getRows(1, rows);
    X_test = X_test / 255.0;
    matrice<float> Y_test = test.getRows(0,1);

    matrice<float> Train = data.iloc(0, cols);
    matrice<float> X_train = Train.getRows(1,rows);
    X_train = X_train / 255.0;
    matrice<float> Y_train = Train.getRows(0,1);

    for(int i = 0; i < 500; i++)
    {
        vec(matrice<float>) results = n.forward(X_train);
        vec(matrice<float>) dds = n.backward_prop(results, X_train, Y_train);
        n.update_params(dds, .01);
        if((i+1) % 10 == 0)
            std::cout << "Iteration: " << (i + 1) << std::endl;
    }
    std::ofstream file("Result.txt");
    int num_correct = 0;
    for(int j = 0; j < 200; j++)
    {
        matrice<float> x(784, 1);
        for(int i = 0; i < 784; i++)
            x[i][0] = X_train[i][j];
        vec(matrice<float>) pred = n.forward(x);
        file << "Prediction # " << (j + 1) << std::endl;
        file << largest_index(pred[3]) << std::endl;
        file << Y_train[0][j] << std::endl;
        if(largest_index(pred[3]) == Y_train[0][j])
            num_correct++;
    }
    file.close();
    std::cout << "Accuracy: " << (float)num_correct / 200 << std::endl;
    return 0;
    vec(matrice<float>) params = init_params();
    
    for(int i = 0; i < 500; i++)
    {
        vec(matrice<float>) results = forward_prop(params[0], params[1], params[2], params[3], X_train); 
        vec(matrice<float>) dds = backward_prop(results[0], results[1], results[2], results[3], params[0], params[2], X_train, Y_train);
        params = update_params(params[0], params[1], params[2], params[3], dds[0], dds[1], dds[2], dds[3], 0.1);
        if((i+1) % 10 == 0)
            std::cout << "Iteration: " << (i + 1) << std::endl;
    }

    // std::ofstream file("Result.txt");
    // int num_correct = 0;
    // for(int j = 0; j < 200; j++)
    // {
    //     matrice<float> x(784, 1);
    //     for(int i = 0; i < 784; i++)
    //         x[i][0] = X_train[i][j];
    //     vec(matrice<float>) pred = forward_prop(params[0], params[1], params[2], params[3], x);
    //     file << "Prediction # " << (j + 1) << std::endl;
    //     file << largest_index(pred[3]) << std::endl;
    //     file << Y_train[0][j] << std::endl;
    //     if(largest_index(pred[3]) == Y_train[0][j])
    //         num_correct++;
    // }
    // file.close();
    // std::cout << "Accuracy: " << (float)num_correct / 200 << std::endl;
}