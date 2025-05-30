#ifndef MATRICE_H
#define MATRICE_H
#include <vector>
#include <iostream>

template <typename T>
class matrice
{
private:
    int row, col;
public:
    std::vector<std::vector<T>> matrix;
    int getNumRows();
    int getNumCols();

    matrice(int row, int col);
    matrice(std::vector<std::vector<T>>& matrix);
    matrice(std::vector<T>& matrix);
    matrice(const std::vector<T>& matrix);
    matrice();
    void toString();
    void addRow(std::vector<T> row);
    void removeRow();
    std::vector<T>& operator[](int index);
    const std::vector<T>& operator[](int index) const;
    std::vector<T> iloc(int index);
    matrice<T> iloc(int start, int end);
    matrice<T> getRows(int start, int end);
    matrice<T> operator-(matrice<T>& inp);
    matrice<T> operator-(T inp);
    matrice<T> operator/(T inp);
    matrice<T> operator*(T inp);
    matrice<T> operator+(matrice<T>& inp);
    void operator=(matrice<T>& inp);
    void operator=(const matrice<T>& inp);
    void operator=(std::vector<std::vector<T>>& inp);
    void operator=(std::vector<T>& inp);
    void update();
    matrice<T> operator-(const matrice<T>& inp);
    matrice<T> operator*(matrice<T>& inp);
    matrice<T> dot(matrice<T>& inp);
    matrice<T> dot(const matrice<T>& inp);
    matrice<T> operator*(const matrice<T>& inp);
    matrice<T> transpose();
    T sum();
    T max();
};

#include "matrice.cpp"
#endif