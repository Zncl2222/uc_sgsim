/**
 * @file native_matrix.h
 * @brief Contiguous row-major matrices and stable SPD solvers.
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_NATIVE_MATRIX_H_
#define UC_SGSIM_C_CORE_INCLUDE_NATIVE_MATRIX_H_

#include <stddef.h>

#include "native_array.h"

typedef struct {
    double* data;
    size_t rows;
    size_t columns;
    size_t stride;
} sgsim_double_matrix_t;

/** Allocate a zeroed, contiguous row-major matrix. */
sgsim_numeric_status_t sgsim_double_matrix_init(
    sgsim_double_matrix_t* matrix,
    size_t rows,
    size_t columns);

/** Release matrix storage. NULL-safe and idempotent. */
void sgsim_double_matrix_free(sgsim_double_matrix_t* matrix);

/** Bounds-checked element access for non-hot-path callers. */
sgsim_numeric_status_t sgsim_double_matrix_get(
    const sgsim_double_matrix_t* matrix,
    size_t row,
    size_t column,
    double* value);
sgsim_numeric_status_t sgsim_double_matrix_set(
    sgsim_double_matrix_t* matrix,
    size_t row,
    size_t column,
    double value);

/**
 * Factor the leading order-by-order symmetric positive-definite block.
 *
 * The lower triangle is consumed and a lower Cholesky factor is written to
 * factor. Ill-conditioned covariance matrices are retried with a bounded,
 * scale-aware diagonal regularizer. The applied value is written to jitter.
 */
sgsim_numeric_status_t sgsim_cholesky_factorize_regularized(
    const sgsim_double_matrix_t* matrix,
    sgsim_double_matrix_t* factor,
    size_t order,
    double* jitter);

/** Solve L L^T x = b using caller-owned work storage. */
sgsim_numeric_status_t sgsim_cholesky_solve(
    const sgsim_double_matrix_t* factor,
    const double* right_hand_side,
    double* work,
    double* solution,
    size_t order);

/** Dot product and x^T A x helpers used by the kriging hot path. */
sgsim_numeric_status_t sgsim_vector_dot(
    const double* left,
    const double* right,
    size_t length,
    double* result);
sgsim_numeric_status_t sgsim_symmetric_quadratic_form(
    const sgsim_double_matrix_t* matrix,
    const double* vector,
    size_t order,
    double* result);

#endif  // UC_SGSIM_C_CORE_INCLUDE_NATIVE_MATRIX_H_
