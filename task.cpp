#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#define N1 1024
#define M2 1024
using type = int;
type* matrix_multiplication_base(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	//#pragma omp parallel for //shared(matr1, matr2, res_matr) reduction (+:temp)
	clock_t st = clock();
#pragma omp parallel for
	for (size_t j = 0; j < m2; ++j) {
		for (size_t i = 0; i < n1; ++i) {
			type temp = 0;
			for (size_t k = 0; k < m1; ++k)
				temp += matr1[i * m1 + k] * matr2[k * m2 + j];
			res_matr[i * m2 + j] = temp;
		}
	}
	std::cout << "\nmatrix_multiplication_base\nTIME: " << (clock() - st) / (double)CLOCKS_PER_SEC << '\n';
	return res_matr;
}

type* matrix_multiplication_sqr_block(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	for (size_t i = 0; i < n1; i++)
		for (size_t j = 0; j < m2; j++)
			res_matr[i * m2 + j] = 0;

	clock_t st = clock();
#pragma omp parallel for
	for (int ib = 0; ib < n1; ib += block_sz)
		for (int kb = 0; kb < m1; kb += block_sz)
			for (int jb = 0; jb < m2; jb += block_sz)
				for (int i = ib; i < ib + block_sz; ++i)
					for (int k = kb; k < kb + block_sz; ++k)
						for (int j = jb; j < jb + block_sz; ++j)
							res_matr[i * m2 + j] += matr1[i * m1 + k] * matr2[k * m2 + j];
	std::cout << "\nmatrix_multiplication_sqr_block\nTIME: " << (clock() - st) / (double)CLOCKS_PER_SEC << '\n';

	return res_matr;
}

type* matrix_multiplication_rect_block(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	for (size_t i = 0; i < n1; i++)
		for (size_t j = 0; j < m2; j++)
			res_matr[i * m2 + j] = 0;

	clock_t st = clock();
#pragma omp parallel for
	for (int jk = 0; jk < n1; jk += block_sz)
		for (int ik = 0; ik < m1; ik += block_sz)
			for (int j = jk; j < jk + block_sz; ++j)
				for (int k = ik; k < ik + block_sz; ++k)
					for (int i = 0; i < m2; ++i)
						res_matr[j * m2 + i] += matr1[j * m1 + k] * matr2[k * m2 + i];
	std::cout << "\nmatrix_multiplication_rect_block\nTIME: " << (clock() - st) / (double)CLOCKS_PER_SEC << '\n';

	return res_matr;
}

type* generate_matrix(const size_t N, const size_t M) {
	type* matrix = new type[N * M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			matrix[i * M + j] = rand() % 1009;

	return matrix;
}

void destructor(type* matrix) {
	if (matrix == nullptr) return;
	delete[]matrix;
	matrix = nullptr;
}

int main()
{
	srand(time(NULL));
	//..........................................................
	type* matr1 = generate_matrix(N1, M2);
	type* matr2 = generate_matrix(N1, M2);
	type* ans = matrix_multiplication_rect_block(matr1, N1, M2, matr2, N1, M2, 32);
	type* ans1 = matrix_multiplication_sqr_block(matr1, N1, M2, matr2, N1, M2, 256);
	type* ans2 = matrix_multiplication_base(matr1, N1, M2, matr2, N1, M2);

	for (size_t i = 0; i < N1; ++i)
		for (size_t j = 0; j < M2; ++j) {
			if (ans[i * M2 + j] != ans2[i * M2 + j]) {
				std::cout << "FAILED RECTANGLE BLOCK\n";
				return 0;
			}
			if (ans[i * M2 + j] != ans1[i * M2 + j]) {
				std::cout << "FAILED SQUARE BLOCK\n";
				return 0;
			}
		}
	std::cout << "PASSED\n";

	destructor(matr1);
	destructor(matr2);
	destructor(ans);
	destructor(ans1);
	destructor(ans2);
	return 0;
}





