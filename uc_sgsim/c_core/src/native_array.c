/**
 * @file native_array.c
 * @brief Checked fixed-size allocation used by the native numerical core.
 */

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "../include/native_array.h"

int sgsim_size_multiply(size_t a, size_t b, size_t* product) {
    if (product == NULL) {
        return 0;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        *product = 0;
        return 0;
    }
    *product = a * b;
    return 1;
}

static sgsim_numeric_status_t allocate_zeroed(
    void** destination,
    size_t length,
    size_t element_size) {
    size_t byte_count;
    if (destination == NULL || element_size == 0) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *destination = NULL;
    if (!sgsim_size_multiply(length, element_size, &byte_count)) {
        return SGSIM_NUMERIC_SIZE_OVERFLOW;
    }
    if (byte_count == 0) {
        return SGSIM_NUMERIC_OK;
    }
    *destination = calloc(length, element_size);
    return *destination == NULL
        ? SGSIM_NUMERIC_ALLOCATION_FAILED
        : SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_int_array_init(
    sgsim_int_array_t* array,
    size_t length) {
    if (array == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *array = (sgsim_int_array_t){0};
    void* data = NULL;
    sgsim_numeric_status_t status = allocate_zeroed(
        &data,
        length,
        sizeof(*array->data));
    if (status == SGSIM_NUMERIC_OK) {
        array->data = data;
        array->length = length;
    }
    return status;
}

sgsim_numeric_status_t sgsim_double_array_init(
    sgsim_double_array_t* array,
    size_t length) {
    if (array == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *array = (sgsim_double_array_t){0};
    void* data = NULL;
    sgsim_numeric_status_t status = allocate_zeroed(
        &data,
        length,
        sizeof(*array->data));
    if (status == SGSIM_NUMERIC_OK) {
        array->data = data;
        array->length = length;
    }
    return status;
}

void sgsim_int_array_free(sgsim_int_array_t* array) {
    if (array == NULL) {
        return;
    }
    free(array->data);
    *array = (sgsim_int_array_t){0};
}

void sgsim_double_array_free(sgsim_double_array_t* array) {
    if (array == NULL) {
        return;
    }
    free(array->data);
    *array = (sgsim_double_array_t){0};
}

sgsim_numeric_status_t sgsim_int_array_iota(sgsim_int_array_t* array) {
    if (array == NULL || (array->length != 0 && array->data == NULL)) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    if (array->length > (size_t)INT_MAX + 1U) {
        return SGSIM_NUMERIC_SIZE_OVERFLOW;
    }
    for (size_t index = 0; index < array->length; index++) {
        array->data[index] = (int)index;
    }
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_int_array_get(
    const sgsim_int_array_t* array,
    size_t index,
    int* value) {
    if (array == NULL || value == NULL || index >= array->length
        || array->data == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *value = array->data[index];
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_int_array_set(
    sgsim_int_array_t* array,
    size_t index,
    int value) {
    if (array == NULL || index >= array->length || array->data == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    array->data[index] = value;
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_double_array_get(
    const sgsim_double_array_t* array,
    size_t index,
    double* value) {
    if (array == NULL || value == NULL || index >= array->length
        || array->data == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    *value = array->data[index];
    return SGSIM_NUMERIC_OK;
}

sgsim_numeric_status_t sgsim_double_array_set(
    sgsim_double_array_t* array,
    size_t index,
    double value) {
    if (array == NULL || index >= array->length || array->data == NULL) {
        return SGSIM_NUMERIC_INVALID_ARGUMENT;
    }
    array->data[index] = value;
    return SGSIM_NUMERIC_OK;
}
