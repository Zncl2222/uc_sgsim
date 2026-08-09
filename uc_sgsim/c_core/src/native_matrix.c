/**
 * @file native_matrix.c
 * @brief Checked contiguous matrices and Cholesky algorithms.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "../include/native_matrix.h"

static int matrix_supports_order(
    const sgsim_double_matrix_t* matrix,
    size_t order) {
    if (matrix == NULL
        || order == 0
        || matrix->data == NULL
        || matrix->rows < order
        || matrix->columns < order
        || matrix->stride < matrix->columns) {
        return 0;
    }
    size_t last_row_offset;
    return sgsim_size_multiply(order - 1U, matrix->stride, &last_row_offset)
        && last_row_offset <= SIZE_MAX - (order - 1U);
}

static int matrix_offset(
    const sgsim_double_matrix_t* matrix,
    size_t row,
    size_t column,
    size_t* offset) {
    size_t row_offset;
    if (matrix == NULL || offset == NULL
        || matrix->data == NULL
        || row >= matrix->rows
        || column >= matrix->columns
        || matrix->stride < matrix->columns
        || !sgsim_size_multiply(row, matrix->stride, &row_offset)
        || row_offset > SIZE_MAX - column) {
        return 0;
    }
    *offset = row_offset + column;
    return 1;
}

sgsim_numeric_status_t sgsim_double_matrix_init(
    sgsim_double_matrix_t* matrix,
    size_t rows,
    size_t columns) {
    size_t element_count;
    size_t byte_count;
    if (matrix == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *matrix = (sgsim_double_matrix_t){0};
    if (!sgsim_size_multiply(rows, columns, &element_count)) {
        return SGSIM_NUMERIC_SIZE_OVERFLOW;
    }
    if (!sgsim_size_multiply(
            element_count,
            sizeof(*matrix->data),
            &byte_count)) {
        return SGSIM_NUMERIC_SIZE_OVERFLOW;
    }
    if (element_count == 0) {
        matrix->rows = rows;
        matrix->columns = columns;
        matrix->stride = columns;
        return SGSIM_NUMERIC_OK;
    }
    matrix->data = calloc(element_count, sizeof(*matrix->data));
    if (matrix->data == NULL) {
        return SGSIM_NUMERIC_ALLOCATION_FAILED;
    }
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->stride = columns;
    return SGSIM_NUMERIC_OK;
}

void sgsim_double_matrix_free(sgsim_double_matrix_t* matrix) {
    if (matrix == NULL) {
        return;
    }
    free(matrix->data);
    *matrix = (sgsim_double_matrix_t){0};
}

sgsim_numeric_status_t sgsim_double_matrix_get(
    const sgsim_double_matrix_t* matrix,
    size_t row,
    size_t column,
    double* value) {
    size_t offset;
    if (value == NULL || !matrix_offset(matrix, row, column, &offset)) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *value = matrix->data[offset];
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_double_matrix_set(
    sgsim_double_matrix_t* matrix,
    size_t row,
    size_t column,
    double value) {
    size_t offset;
    if (!matrix_offset(matrix, row, column, &offset)) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    matrix->data[offset] = value;
    return SGSIM_NUMERIC_OK;
}

static sgsim_numeric_status_t cholesky_attempt(
    const sgsim_double_matrix_t* matrix,
    sgsim_double_matrix_t* factor,
    size_t order,
    double jitter,
    double minimum_pivot) {
    const size_t input_stride = matrix->stride;
    const size_t factor_stride = factor->stride;

    for (size_t row = 0; row < order; row++) {
        for (size_t column = 0; column <= row; column++) {
            double value = matrix->data[row * input_stride + column];
            if (!isfinite(value)) {
                return SGSIM_NUMERIC_NONFINITE_VALUE;
            }
            factor->data[row * factor_stride + column] =
                row == column ? value + jitter : value;
        }
    }

    for (size_t row = 0; row < order; row++) {
        for (size_t column = 0; column <= row; column++) {
            double value = factor->data[row * factor_stride + column];
            const double* row_data = &factor->data[row * factor_stride];
            const double* column_data = &factor->data[column * factor_stride];
            for (size_t index = 0; index < column; index++) {
                value -= row_data[index] * column_data[index];
            }

            if (row == column) {
                if (!isfinite(value) || value <= minimum_pivot) {
                    return SGSIM_NUMERIC_NOT_POSITIVE_DEFINITE;
                }
                factor->data[row * factor_stride + column] = sqrt(value);
            } else {
                double diagonal = factor->data[column * factor_stride + column];
                value /= diagonal;
                if (!isfinite(value)) {
                    return SGSIM_NUMERIC_NONFINITE_VALUE;
                }
                factor->data[row * factor_stride + column] = value;
            }
        }
    }
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_cholesky_factorize_regularized(
    const sgsim_double_matrix_t* matrix,
    sgsim_double_matrix_t* factor,
    size_t order,
    double* jitter) {
    if (!matrix_supports_order(matrix, order)
        || !matrix_supports_order(factor, order)
        || matrix->data == factor->data
        || jitter == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }

    double scale = 0.0;
    for (size_t index = 0; index < order; index++) {
        double diagonal = matrix->data[index * matrix->stride + index];
        if (!isfinite(diagonal)) {
            return SGSIM_NUMERIC_NONFINITE_VALUE;
        }
        scale = fmax(scale, fabs(diagonal));
    }
    double minimum_pivot = fmax(
        scale * DBL_EPSILON * (double)order * 16.0,
        DBL_MIN);

    *jitter = 0.0;
    sgsim_numeric_status_t status = cholesky_attempt(
        matrix,
        factor,
        order,
        0.0,
        minimum_pivot);
    if (status == SGSIM_NUMERIC_OK || status == SGSIM_NUMERIC_NONFINITE_VALUE) {
        return status;
    }

    double candidate_jitter = minimum_pivot;
    for (int attempt = 0; attempt < 7; attempt++) {
        status = cholesky_attempt(
            matrix,
            factor,
            order,
            candidate_jitter,
            minimum_pivot);
        if (status == SGSIM_NUMERIC_OK) {
            *jitter = candidate_jitter;
            return status;
        }
        if (status == SGSIM_NUMERIC_NONFINITE_VALUE) {
            return status;
        }
        candidate_jitter *= 10.0;
    }
    return SGSIM_NUMERIC_NOT_POSITIVE_DEFINITE;
}

sgsim_numeric_status_t sgsim_cholesky_solve(
    const sgsim_double_matrix_t* factor,
    const double* right_hand_side,
    double* work,
    double* solution,
    size_t order) {
    if (!matrix_supports_order(factor, order)
        || right_hand_side == NULL || work == NULL || solution == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }

    for (size_t row = 0; row < order; row++) {
        double value = right_hand_side[row];
        const double* factor_row = &factor->data[row * factor->stride];
        for (size_t column = 0; column < row; column++) {
            value -= factor_row[column] * work[column];
        }
        double diagonal = factor_row[row];
        if (!isfinite(value) || !isfinite(diagonal) || diagonal <= 0.0) {
            return SGSIM_NUMERIC_NONFINITE_VALUE;
        }
        work[row] = value / diagonal;
    }

    for (size_t row = order; row-- > 0;) {
        double value = work[row];
        for (size_t column = row + 1; column < order; column++) {
            value -= factor->data[column * factor->stride + row] * solution[column];
        }
        value /= factor->data[row * factor->stride + row];
        if (!isfinite(value)) {
            return SGSIM_NUMERIC_NONFINITE_VALUE;
        }
        solution[row] = value;
    }
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_vector_dot(
    const double* left,
    const double* right,
    size_t length,
    double* result) {
    if (left == NULL || right == NULL || result == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    double sum = 0.0;
    for (size_t index = 0; index < length; index++) {
        sum += left[index] * right[index];
    }
    if (!isfinite(sum)) {
        return SGSIM_NUMERIC_NONFINITE_VALUE;
    }
    *result = sum;
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_symmetric_quadratic_form(
    const sgsim_double_matrix_t* matrix,
    const double* vector,
    size_t order,
    double* result) {
    if (!matrix_supports_order(matrix, order)
        || vector == NULL || result == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    double sum = 0.0;
    for (size_t row = 0; row < order; row++) {
        const double* matrix_row = &matrix->data[row * matrix->stride];
        sum += vector[row] * matrix_row[row] * vector[row];
        for (size_t column = 0; column < row; column++) {
            sum += 2.0 * vector[row] * matrix_row[column] * vector[column];
        }
    }
    if (!isfinite(sum)) {
        return SGSIM_NUMERIC_NONFINITE_VALUE;
    }
    *result = sum;
    return SGSIM_NUMERIC_OK;
}
