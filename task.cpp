#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>


#define N1 1003
#define M1 2047

using type = double;

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
	for (int ib = 0; ib < n1; ib += block_sz_n)
		for (int kb = 0; kb < m1; kb += block_sz_m)
			for (int jb = 0; jb < m2; jb += block_sz_n)
				for (int i = ib; i < std::min(ib + block_sz_n, n1); ++i)
					for (int k = kb; k < std::min(kb + block_sz_m, m1); ++k)
#pragma omp simd
						for (int j = jb; j < std::min(jb + block_sz_n, m2); ++j)
							res_matr[i * m2 + j] += matr1[i * m1 + k] * matr2[k * m2 + j];

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nmatrix_multiplication_block\nTIME: ";
	std::chrono::duration<double> duration = end - start;
	std::cout << duration.count() << std::endl;
	return res_matr;
}

type* matrix_multiplication_line_block(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int jk = 0; jk < n1; jk += block_sz)
		for (int ik = 0; ik < m1; ik += block_sz)
			for (int j = jk; j < std::min(jk + block_sz, n1); ++j)
				for (int k = ik; k < std::min(ik + block_sz, m1); ++k)
#pragma omp simd
					for (int i = 0; i < m2; ++i)
						res_matr[j * m2 + i] += matr1[j * m1 + k] * matr2[k * m2 + i];
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nmatrix_multiplication_line_block\nTIME: ";
	std::chrono::duration<double> duration = end - start;
	std::cout << duration.count() << std::endl;
	return res_matr;
}

type* generate_matrix(const size_t N, const size_t M) {
	double MAX = FLT_MAX, MIN = -MAX;
	type* matrix = new type[N * M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			matrix[i * M + j] = MIN + (double)rand() / (double)(RAND_MAX / (MAX - MIN));
	//matrix[i * M + j] = rand() % 200003;

	return matrix;
}

void destructor(type* matrix) {
	if (matrix == nullptr) return;
	delete[]matrix;
	matrix = nullptr;
}

void launch(int k) {
	srand(time(NULL));
	type* matr1 = generate_matrix(N1, M1);
	type* matr2 = generate_matrix(M1, M1);
	type* ans;
	switch (k) {
	case 0: {
		ans = matrix_multiplication_line_block(matr1, N1, M1, matr2, M1, M1, 128);
		break;
	}
	case 1: {
		ans = matrix_multiplication_block(matr1, N1, M1, matr2, M1, M1, 128, 256);
		break;
	}
	case 2: {
		ans = matrix_multiplication_base(matr1, N1, M1, matr2, M1, M1);
		break;
	}
	}
	destructor(matr1);
	destructor(matr2);
	destructor(ans);
}

int main()
{
	//.........................................................
	for (size_t i = 0; i < 3; ++i) {
		launch(i);
	}
	//for (size_t i = 0; i < N1; ++i)
	//	for (size_t j = 0; j < M1; ++j) {
	//		if (ans[i * M1 + j] != ans2[i * M1 + j]) {
	//			std::cout << "FAILED LINE BLOCK\n";
	//			return 0;
	//		}
	//		if (ans[i * M1 + j] != ans1[i * M1 + j]) {
	//			std::cout << "FAILED BLOCK\n";
	//			return 0;
	//		}
	//	}
	//std::cout << "PASSED\n";

	return 0;
}





