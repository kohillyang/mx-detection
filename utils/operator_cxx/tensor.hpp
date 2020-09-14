/*
 * tensor.hpp
 *
 *  Created on: Sep 4, 2020
 *      Author: kk
 */

#ifndef UTILS_OPERATOR_CXX_PAALOSS_TENSOR_HPP_
#define UTILS_OPERATOR_CXX_PAALOSS_TENSOR_HPP_
#include "mobula_op.h"
#define TENSOR_DEBUG 1
#if TENSOR_DEBUG
#define BOUND_CHECK(x) assert(x)
#else
#define BOUND_CHECK(x)
#endif


template <typename T>
class Tensor3D{
public:

	int shape0;
	int shape1;
	int shape2;
	T *data;
	MOBULA_DEVICE Tensor3D(T *data,  int shape0,  int shape1,  int shape2){
		this->data = data;
		this->shape0 = shape0;
		this->shape1 = shape1;
		this->shape2 = shape2;
	}
	MOBULA_DEVICE T& operator()(int idx_dim0, int idx_dim1, int idx_dim2){
		BOUND_CHECK(idx_dim0 < shape0);
		BOUND_CHECK(idx_dim1 < shape1);
		BOUND_CHECK(idx_dim2 < shape2);
		return data[idx_dim0 * ( shape1 * shape2) + idx_dim1 * shape2 + idx_dim2];
	}
};

template <typename T>
class Tensor4D{
public:

	int shape0;
	int shape1;
	int shape2;
	int shape3;
	T *data;
	MOBULA_DEVICE Tensor4D(T *data, int shape0, int shape1, int shape2, int shape3){
		this->data = data;
		this->shape0 = shape0;
		this->shape1 = shape1;
		this->shape2 = shape2;
		this->shape3 = shape3;
	}
	MOBULA_DEVICE T& operator()(int idx_dim0, int idx_dim1, int idx_dim2, int idx_dim3){
		BOUND_CHECK(idx_dim0 < shape0);
		BOUND_CHECK(idx_dim1 < shape1);
		BOUND_CHECK(idx_dim2 < shape2);
		BOUND_CHECK(idx_dim3 < shape3);

		return data[idx_dim0 * ( shape1 * shape2 * shape3) + idx_dim1 * (shape2 * shape3) + idx_dim2 * shape3 + idx_dim3];
	}
};


template <typename T>
class Tensor5D{
public:
	int shape0;
	int shape1;
	int shape2;
	int shape3;
	int shape4;

	T *data;
	MOBULA_DEVICE Tensor5D(T *data,  int shape0,  int shape1,  int shape2,  int shape3,  int shape4){
		this->data = data;
		this->shape0 = shape0;
		this->shape1 = shape1;
		this->shape2 = shape2;
		this->shape3 = shape3;
		this->shape4 = shape4;
	}
	MOBULA_DEVICE T& operator()(int idx_dim0, int idx_dim1, int idx_dim2, int idx_dim3, int idx_dim4){
		BOUND_CHECK(idx_dim0 < shape0);
		BOUND_CHECK(idx_dim1 < shape1);
		BOUND_CHECK(idx_dim2 < shape2);
		BOUND_CHECK(idx_dim3 < shape3);
		BOUND_CHECK(idx_dim4 < shape4);
		return data[idx_dim0 * ( shape1 * shape2 * shape3 * shape4) + idx_dim1 * (shape2 * shape3 * shape4) + idx_dim2 * (shape3 * shape4) + idx_dim3 * shape4 + idx_dim4];
	}
};

template <typename T>
class Tensor6D{
public:
	int shape0;
	int shape1;
	int shape2;
	int shape3;
	int shape4;
	int shape5;

	T *data;
	MOBULA_DEVICE Tensor6D(T *data,  int shape0,  int shape1,  int shape2,  int shape3,  int shape4,  int shape5){
		this->data = data;
		this->shape0 = shape0;
		this->shape1 = shape1;
		this->shape2 = shape2;
		this->shape3 = shape3;
		this->shape4 = shape4;
		this->shape5 = shape5;

	}
	MOBULA_DEVICE T& operator()(int idx_dim0, int idx_dim1, int idx_dim2, int idx_dim3, int idx_dim4, int idx_dim5){
		BOUND_CHECK(idx_dim0 < shape0);
		BOUND_CHECK(idx_dim1 < shape1);
		BOUND_CHECK(idx_dim2 < shape2);
		BOUND_CHECK(idx_dim3 < shape3);
		BOUND_CHECK(idx_dim4 < shape4);
		BOUND_CHECK(idx_dim5 < shape5);
		return data[idx_dim0 * (shape1 * shape2 * shape3 * shape4 * shape5) +
					idx_dim1 * (shape2 * shape3 * shape4 * shape5) +
					idx_dim2 * (shape3 * shape4 * shape5) +
					idx_dim3 * (shape4 * shape5) +
					idx_dim4 * shape5 +
					idx_dim5];
	}
};

#endif /* UTILS_OPERATOR_CXX_PAALOSS_TENSOR_HPP_ */
