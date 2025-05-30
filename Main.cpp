#include <iostream>
#include "matrice.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Network.h"
#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>

std::vector<std::vector<float>> loadCSV(std::string fileName, bool header=true, int maxRows=1000, int row_number=0)
{
    std::vector<std::vector<float>> data;
    std::ifstream file(fileName);
    std::string line;
    if(header)
        getline(file, line);
    int i = 0;
    while(i < row_number && getline(file, line))
        i++;
    i = 0;
    while(getline(file, line) && i < maxRows)
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

int largest_index(matrice<float>& input)
{
    int index = 0;
    for(int i = 0; i < input.getNumRows(); i++)
        index = input[i][0] > input[index][0]? i: index;
    return index;
}

int main()
{ 
    srand(time(0)+45);
    Network n;
    n.addLayer(784, 0, 0);
    n.addLayer(10, ReLU, ReLU_derive);
    n.addLayer(10, softmax, 0);

    n.setRandomization(randomize_matrix);
    n.applyRandomzation(1);
    n.applyRandomzation(2);

    int counter = 0;
    for(int i = 0; i < n.layers.size()-1; i++)
    {
        Layer& layer = n.layers[i+1];
        layer.weight.matrix = loadCSV("Network.txt", false, layer.weight.getNumRows(), counter);
        counter += layer.weight.getNumRows();
        layer.bias.matrix = loadCSV("Network.txt", false, layer.bias.getNumRows(), counter);
        counter += layer.bias.getNumRows();
    }
    
    std::string fileName = "mnist/train.csv";
    std::vector<std::vector<float>> df = loadCSV(fileName, true, 1000);
    std::vector<std::vector<float>> df_test = loadCSV("image.csv", false);
    matrice<float> x_test(df_test);
    x_test = x_test.transpose();
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

    if(false)
    {
        for(int i = 0; i < 500; i++)
        {
            vec(matrice<float>) results = n.forward(X_train);
            vec(matrice<float>) dds = n.backward_prop(results, X_train, Y_train);
            n.update_params(dds, .1);
            if((i+1) % 10 == 0)
                std::cout << "Iteration: " << (i + 1) << std::endl;
        }
    }
    
    if(false)
    {
        std::ofstream network("Network.txt");
        for(Layer& layer: n.layers)
        {
            std::cout << layer.weight.getNumRows() << std::endl;
            std::cout << layer.bias.getNumRows() << std::endl;
            for(int i = 0; i < layer.weight.getNumRows(); i++)
            {
                for(int j = 0; j < layer.weight.getNumCols(); j++)
                    network << layer.weight[i][j] << ",";
                network << std::endl;
            }
            for(int i = 0; i < layer.bias.getNumRows(); i++)
            {
                for(int j = 0; j < layer.bias.getNumCols(); j++)
                    network << layer.bias[i][j] << ",";
                network << std::endl;
            }
        }
        network.close();
    }
    vec(matrice<float>) pred = n.forward(x_test);
    std::cout << largest_index(pred[3]) << std::endl;
    pred[3].toString();
    
    return 0;
    
}