#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <random>
#include <omp.h>
int N1 = 4096, M1 = 4096, N2 = 4096, M2 = 4096;
const int BLOCK_N = 64, BLOCK_M = 64 * 2;
//#define DEBUG
double t_block[BLOCK_N * BLOCK_M];

#ifdef DEBUG
using type = int;
#else 
using type = double;
#endif

void print(type* matr, const size_t N, const size_t M) {
	std::ofstream out;
	out.open("myfile.txt", std::ios::app);
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			out << matr[i * M + j] << " ";
		}
		out << std::endl;
	}
	out << std::endl;
	out.close();
}

type* matrix_multiplication_base(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2) {
	std::ofstream out;
	out.open("myfile.txt", std::ios::app);
	if (m1 != n2) {
		out << "Incorrect multiplication\n";
		return nullptr;
	}
	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int i = 0; i < n1; ++i)
		for (int k = 0; k < m1; ++k)
#pragma omp simd
			for (size_t j = 0; j < m2; ++j)
				res_matr[i * m2 + j] += matr1[i * m1 + k] * matr2[k * m2 + j];

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	out << duration.count() << std::endl;
	out.close();
	//print(res_matr, n1, m2);
	return res_matr;
}

//------------------------------ simple block algorithm for matrix multiplication ----------------------------------------
type* matrix_multiplication_block\
(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz_n, size_t block_sz_m) {
	std::ofstream out;
	out.open("myfile.txt", std::ios::app);
	if (m1 != n2) {
		out << "Incorrect multiplication\n";
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
	std::chrono::duration<double> duration = end - start;
	out << duration.count() << std::endl;
	out.close();
	return res_matr;
}

//----------------------- rectangle block algorithm for multiplication with transposition inside a block ----------------------------------
type* matrix_multiplication_transpose_rectangle_block\
(type* matr1, int n1, int m1, type* matr2, int n2, int m2, size_t block_n, size_t block_m) {
	std::ofstream out;
	out.open("myfile.txt", std::ios::app);
	if (m1 != n2) {
		out << "Incorrect multiplication\n";
		return nullptr;
	}
	type* res_matr = new type[n1 * m2];
	memset(res_matr, 0, n1 * m2 * sizeof(type));

	auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int jb = 0; jb < m2; jb += block_n) {
		for (int kb = 0; kb < m1; kb += block_m) {
			int up_bm = kb + block_m < m1 ? block_m : m1 - kb;
			int up_bn = jb + block_n < m2 ? block_n : m2 - jb;
			for (int tn = 0; tn < up_bn; ++tn) {
				for (int tm = 0; tm < up_bm; ++tm)
					t_block[tn * block_m + tm] = matr2[(kb + tm) * m2 + (jb + tn)];
			}
			for (int ib = 0; ib < n1; ib += block_n) {
				int up_i = ib + block_n < n1 ? block_n : n1 - ib;
				int up_j = jb + block_n < m2 ? block_n : m2 - jb;
				int up_k = kb + block_m < m1 ? block_m : m1 - kb;
				for (int i = 0; i < up_i; ++i) {
					for (int j = 0; j < up_j; ++j) {
#pragma omp simd
						for (int k = 0; k < up_k; ++k) {
							res_matr[(ib + i) * m2 + jb + j] += matr1[(ib + i) * m1 + kb + k] * t_block[j * block_m + k];
						}

					}
				}
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	out << duration.count() << "\n";
	out.close();
	//print(res_matr, n1, m2);
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
			matrix[i * M + j] = rand() % 2003;
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
		ans = matrix_multiplication_block(matr1, N1, M1, matr2, M1, M1, BLOCK_N, BLOCK_M);
		break;
	}
	case 2: {
		ans = matrix_multiplication_transpose_rectangle_block(matr1, N1, M1, matr2, M1, M1, BLOCK_N, BLOCK_M);
		break;
	}
	}
	destructor(matr1);
	destructor(matr2);
	destructor(ans);
}

void destructor_for_local_launch(type* matr1, type* matr2, type* ans, type* ans1, type* ans2) {
	destructor(matr1);
	destructor(matr2);
	destructor(ans);
	destructor(ans1);
	destructor(ans2);
}

void local_launch() {
	srand(time(NULL));
	std::ofstream out;
	out.open("myfile.txt", std::ios::app);
	type* matr1 = generate_int_matrix(N1, M1);
	type* matr2 = generate_int_matrix(M1, M1);
	out.close();
	type* ans = matrix_multiplication_base(matr1, N1, M1, matr2, M1, M1);
	type* ans1 = matrix_multiplication_block(matr1, N1, M1, matr2, M1, M1, BLOCK_N, BLOCK_M);
	type* ans2 = matrix_multiplication_transpose_rectangle_block(matr1, N1, M1, matr2, M1, M1, BLOCK_N, BLOCK_M);
	out.open("myfile.txt", std::ios::app);
	for (size_t i = 0; i < N1; ++i)
		for (size_t j = 0; j < M1; ++j) {
			if (ans[i * M1 + j] != ans1[i * M1 + j]) {
				out << i << " " << j << " FAILED: SQR BLOCK\n";
				destructor_for_local_launch(matr1, matr2, ans, ans1, ans2);
				return;
			}
			if (ans[i * M1 + j] != ans2[i * M1 + j]) {
				out << i << " " << j << " FAILED: TRANSPOSE RECTANGLE BLOCK\n";
				destructor_for_local_launch(matr1, matr2, ans, ans1, ans2);
				return;
			}
		}
	out << "PASSED\n";
	out.close();
	destructor_for_local_launch(matr1, matr2, ans, ans1, ans2);
}
int main(int argc, char** argv) {
	std::ofstream out;
	out.open("myfile.txt");
	out << "******************************* Start test **************************************\n";
	//N1 = std::stoi(std::string(argv[1])), M1 = std::stoi(std::string(argv[2]));
	//N2 = std::stoi(std::string(argv[3])), M2 = std::stoi(std::string(argv[4]));
	//BLOCK_N = std::stoi(std::string(argv[5])), BLOCK_M = std::stoi(std::string(argv[6]));
	out << "N1 = " << N1 << " M1 = " << M1 << " N2 = " << N2 << " M2 = " << M2 << std::endl;
	out << "BLOCK_N = " << BLOCK_N << " BLOCK_M = " << BLOCK_M << std::endl;
#ifdef DEBUG
	out.close();
	local_launch();
#else
	out.close();
	for (size_t i = 0; i < 3; ++i) {
		out.open("myfile.txt", std::ios::app);
		switch (i) {
		case 0: {
			out << "\nmatrix_multiplication_base()\nTIME:\n";
			break;
		}
		case 1: {
			out << "\nmatrix_multiplication_block()\nTIME:\n";
			break;
		}
		case 2: {
			out << "\nmatrix_multiplication_transpose_rectangle_block()\nTIME:\n";
			break;
		}
		}

		out << "\nBLOCK_N = " << BLOCK_N << " BLOCK_M = " << BLOCK_M << std::endl;
		for (int cnt = 0; cnt < 3; ++cnt)
			launch(i);
		out.close();
	}
#endif
	out.open("myfile.txt", std::ios::app);
	out << "******************************* Finish test **************************************\n";
	out.close();
	return 0;
}