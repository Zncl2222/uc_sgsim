/**
 * @file matrix_tools.h
 * @brief Matrix Tools Library
 *
 * This header file provides declarations for functions related to matrix operations and array manipulations.
 * The functions in this library assist in operations like LU decomposition, inverse solving, array generation,
 * distance calculations, matrix transformations, and data saving.
 *
 * Copyright (c) 2022-2023 Zncl2222
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_MATRIX_TOOLS_H_
#define UC_SGSIM_C_CORE_INCLUDE_MATRIX_TOOLS_H_

/**
 * @brief Perform LU decomposition on a matrix.
 *
 * This function performs LU decomposition on a square matrix and returns the lower (L) and upper (U) triangular matrices.
 *
 * @param mat A pointer to the input square matrix.
 * @param l A pointer to the lower triangular matrix (output).
 * @param u A pointer to the upper triangular matrix (output).
 * @param n The size of the square matrix (n x n).
 */
void lu_decomposition(double** mat, double** l, double** u, int n);

/**
 * @brief Solve a linear system using LU decomposition.
 *
 * This function solves a linear system of equations using LU decomposition and returns the solution in the `result` array.
 *
 * @param mat A pointer to the coefficient matrix (n x n).
 * @param array A pointer to the right-hand side vector (length n).
 * @param result A pointer to the result vector (output).
 * @param n The size of the system and matrices (n x n).
 */
void lu_inverse_solver(double** mat, const double* array, double* result, int n);

/**
 * @brief Generate an integer array with values from 0 to x-1.
 *
 * This function generates an integer array containing values from 0 to x-1.
 *
 * @param x The length of the generated array.
 *
 * @return A pointer to the generated integer array.
 */
int* arange(int x);

/**
 * @brief Generate a double array with values from 0.0 to x-1.
 *
 * This function generates a double array containing values from 0.0 to x-1.
 *
 * @param x The length of the generated array.
 *
 * @return A pointer to the generated double array.
 */
double* d_arange(int x);

/**
 * @brief Calculate pairwise distances between points.
 *
 * This function calculates the pairwise distances between points in a vector and stores them in a square matrix.
 *
 * @param x A pointer to the input vector containing n_dim-dimensional points.
 * @param c A pointer to the square distance matrix (output).
 * @param n_dim The number of dimensions of the points.
 */
void pdist(const double* x, double** c, int n_dim);

/**
 * @brief Transform a 1D array into a square matrix.
 *
 * This function transforms a 1D array of values into a square matrix of size n_dim x n_dim.
 *
 * @param x A pointer to the input 1D array.
 * @param matrix A pointer to the output square matrix.
 * @param n_dim The number of dimensions (size of the square matrix).
 */
void matrixform(const double* x, double**matrix, int n_dim);

/**
 * @brief Save a 1D array to a file.
 *
 * This function saves a 1D array to a file with the specified file header, path, and additional information.
 *
 * @param array A pointer to the input 1D array.
 * @param array_size The size of the input array.
 * @param fhead The file header or name prefix.
 * @param path The directory path where the file will be saved.
 * @param total_n The total number of files in the series.
 * @param curr_n The current file number in the series.
 */
void save_1darray(const double* array, int array_size,
                const char* fhead, const char* path, int total_n, int curr_n);

#endif  // UC_SGSIM_C_CORE_INCLUDE_MATRIX_TOOLS_H_
