#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

#define N1 4096
#define M1 4096
//#define DEBUG

#ifdef DEBUG
using type = int;
#else 
using type = double;
#endif

void print(type* matr, const size_t N, const size_t M) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			std::cout << matr[i * M + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

type* matrix_multiplication_base(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int i = 0; i < n1; ++i)
		for (int k = 0; k < m1; ++k)
#pragma omp simd
			for (int j = 0; j < m2; ++j)
				res_matr[i * m2 + j] += matr1[i * m1 + k] * matr2[k * m2 + j];

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nmatrix_multiplication_base\nTIME: ";
	std::chrono::duration<double> duration = end - start;
	std::cout << duration.count() << std::endl;
	return res_matr;
}

//------------------------------ simple block algorithm for matrix multiplication ----------------------------------------
type* matrix_multiplication_block\
(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz_n, size_t block_sz_m) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}
	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int ib = 0; ib < n1; ib += block_sz_n) {
		int up_i = ib + block_sz_n < n1 ? ib + block_sz_n : n1;
		for (int kb = 0; kb < m1; kb += block_sz_m) {
			int up_k = kb + block_sz_m < m1 ? kb + block_sz_m : m1;
			for (int jb = 0; jb < m2; jb += block_sz_n) {
				int up_j = jb + block_sz_n < m2 ? jb + block_sz_n : m2;
				for (int i = ib; i < up_i; ++i) {
					for (int k = kb; k < up_k; ++k) {
#pragma omp simd
						for (int j = jb; j < up_j; ++j)
							res_matr[i * m2 + j] += matr1[i * m1 + k] * matr2[k * m2 + j];
					}
				}
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nmatrix_multiplication_block\nTIME: ";
	std::chrono::duration<double> duration = end - start;
	std::cout << duration.count() << std::endl;
	return res_matr;
}
//----------------------- square block algorithm for multiplication with transposition inside a block ----------------------------------
type* matrix_multiplication_sqr_block\
(type* matr1, int n1, int m1, type* matr2, int n2, int m2, size_t block) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}
	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int jb = 0; jb < m2; jb += block) {
		for (int kb = 0; kb < m1; kb += block) {
			for (int tn = 0; tn < block; ++tn) {
				for (int tm = tn + 1; tm < block; ++tm)
					std::swap(matr2[(kb + tn) * m2 + (jb + tm)], matr2[(kb + tm) * m2 + (jb + tn)]);
			}
			for (int ib = 0; ib < n1; ib += block)
				for (int i = 0; i < block; ++i)
					for (int j = 0; j < block; ++j) {
#pragma omp simd 
						for (int k = 0; k < block; ++k) {
							res_matr[(ib + i) * m2 + jb + j] += matr1[(ib + i) * m1 + kb + k] * matr2[(kb + j) * m2 + jb + k];
						}

					}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nmatrix_multiplication_sqr_transpose_block\nTIME: ";
	std::chrono::duration<double> duration = end - start;
	std::cout << duration.count() << std::endl;

	//transpose to original matr2
	for (int jb = 0; jb < m2; jb += block) {
		for (int kb = 0; kb < m1; kb += block) {
			for (int tn = 0; tn < block; ++tn) {
				for (int tm = tn + 1; tm < block; ++tm)
					std::swap(matr2[(kb + tn) * m2 + (jb + tm)], matr2[(kb + tm) * m2 + (jb + tn)]);
			}
		}
	}
	return res_matr;
}

type* generate_double_matrix(const size_t N, const size_t M) {
	double MAX = FLT_MAX, MIN = -MAX;
	type* matrix = new type[N * M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			matrix[i * M + j] = MIN + (double)rand() / (double)(RAND_MAX / (MAX - MIN));
	return matrix;
}
type* generate_int_matrix(const size_t N, const size_t M) {
	type* matrix = new type[N * M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			matrix[i * M + j] = rand() % 20003;
	return matrix;
}

void destructor(type* matr) {
	if (matr == nullptr) return;
	delete[](matr);
	matr = nullptr;
}

void launch(int k) {
	srand(time(NULL));
	type* matr1 = generate_double_matrix(N1, M1);
	type* matr2 = generate_double_matrix(M1, M1);
	type* ans;

	switch (k) {
	case 0: {
		ans = matrix_multiplication_base(matr1, N1, M1, matr2, M1, M1);
		break;
	}
	case 1: {
		ans = matrix_multiplication_block(matr1, N1, M1, matr2, M1, M1, 128, 256);
		break;
	}
	case 2: {
		ans = matrix_multiplication_sqr_block(matr1, N1, M1, matr2, M1, M1, 128);
		break;
	}
	}
	destructor(matr1);
	destructor(matr2);
	destructor(ans);
}

void local_launch() {
	srand(time(NULL));
	type* matr1 = generate_int_matrix(N1, M1);
	type* matr2 = generate_int_matrix(M1, M1);

	type* ans = matrix_multiplication_base(matr1, N1, M1, matr2, M1, M1);
	type* ans1 = matrix_multiplication_block(matr1, N1, M1, matr2, M1, M1, 128, 256);
	type* ans2 = matrix_multiplication_sqr_block(matr1, N1, M1, matr2, M1, M1, 128);

	for (size_t i = 0; i < N1; ++i)
		for (size_t j = 0; j < M1; ++j) {
			if (ans[i * M1 + j] != ans1[i * M1 + j]) {
				std::cout << i << " " << j << " FAILED: SQR BLOCK\n";
				destructor(matr1);
				destructor(matr2);
				destructor(ans);
				destructor(ans1);
				destructor(ans2);
				return;
			}
			if (ans[i * M1 + j] != ans2[i * M1 + j]) {
				std::cout << i << " " << j << " FAILED: TRANSPOSE SQR BLOCK\n";
				destructor(matr1);
				destructor(matr2);
				destructor(ans);
				destructor(ans1);
				destructor(ans2);
				return;
			}

		}
	std::cout << "PASSED\n";
	destructor(matr1);
	destructor(matr2);
	destructor(ans);
	destructor(ans1);
	destructor(ans2);
}


int main() {

#ifdef DEBUG
	local_launch();
#else
	for (size_t i = 0; i < 3; ++i)
		launch(i);
#endif

	return 0;
}
