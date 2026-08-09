/**
 * @file native_array.h
 * @brief Fixed-size, checked contiguous storage for the native engine.
 *
 * These containers deliberately do not implement implicit growth. Native
 * simulation workspaces know their final size up front, so fixed allocation
 * keeps ownership explicit, prevents pointer invalidation, and avoids hidden
 * allocations in hot loops.
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_NATIVE_ARRAY_H_
#define UC_SGSIM_C_CORE_INCLUDE_NATIVE_ARRAY_H_

#include <stddef.h>

typedef enum {
    SGSIM_NUMERIC_OK = 0,
    SGSIM_NUMERIC_INVALID_ARGUMENT = 1,
    SGSIM_NUMERIC_SIZE_OVERFLOW = 2,
    SGSIM_NUMERIC_ALLOCATION_FAILED = 3,
    SGSIM_NUMERIC_NOT_POSITIVE_DEFINITE = 4,
    SGSIM_NUMERIC_NONFINITE_VALUE = 5
} sgsim_numeric_status_t;

typedef struct {
    int* data;
    size_t length;
} sgsim_int_array_t;

typedef struct {
    double* data;
    size_t length;
} sgsim_double_array_t;

/** Return non-zero and write the product when a * b is representable. */
int sgsim_size_multiply(size_t a, size_t b, size_t* product);

/** Allocate zero-initialized fixed-size storage. */
sgsim_numeric_status_t sgsim_int_array_init(
    sgsim_int_array_t* array,
    size_t length);
sgsim_numeric_status_t sgsim_double_array_init(
    sgsim_double_array_t* array,
    size_t length);

/** Release owned storage. Both operations are NULL-safe and idempotent. */
void sgsim_int_array_free(sgsim_int_array_t* array);
void sgsim_double_array_free(sgsim_double_array_t* array);

/** Fill an integer array with 0, 1, ..., length - 1. */
sgsim_numeric_status_t sgsim_int_array_iota(sgsim_int_array_t* array);

/** Bounds-checked accessors for code outside validated hot loops. */
sgsim_numeric_status_t sgsim_int_array_get(
    const sgsim_int_array_t* array,
    size_t index,
    int* value);
sgsim_numeric_status_t sgsim_int_array_set(
    sgsim_int_array_t* array,
    size_t index,
    int value);
sgsim_numeric_status_t sgsim_double_array_get(
    const sgsim_double_array_t* array,
    size_t index,
    double* value);
sgsim_numeric_status_t sgsim_double_array_set(
    sgsim_double_array_t* array,
    size_t index,
    double value);

#endif  // UC_SGSIM_C_CORE_INCLUDE_NATIVE_ARRAY_H_
