#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <random>

using type = int;

type* matrix_multiplication(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1*m2];
	//#pragma omp parallel for //shared(matr1, matr2, res_matr) reduction (+:temp)
	for (size_t j = 0; j < m2; ++j) {
		for (size_t i = 0; i < n1; ++i) {
			type temp = 0;
			for (size_t k = 0; k < m1; ++k)
				temp += matr1[i * m1 + k] * matr2[k * m2 + j];
			res_matr[i * m2 + j] = temp;
		}
	}

	return res_matr;
}
type* generate_matrix(const size_t N, const size_t M) {

	type* matrix = new type[N * M];

	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			matrix[i * M + j] = rand() % 4;
			std::cout << matrix[i * M + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	return matrix;
}

void destructor(type* matrix, size_t N, size_t M) {
	if (matrix == nullptr) return;
	delete[]matrix;
	matrix = nullptr;
}

int main()
{
	srand(time(NULL));
//..........................................................
	type* matr1 = generate_matrix(3, 4);
	type* matr2 = generate_matrix(4, 3);
	type* ans = matrix_multiplication(matr1, 3, 4, matr2, 4, 3);
	for (size_t i = 0; i < 3; ++i) {
		for (size_t j = 0; j < 3; ++j)
			std::cout << ans[i * 3 + j] << ' ';
		std::cout << std::endl;
	}

	destructor(matr1, 3, 4);
	destructor(matr2, 4, 3);
	destructor(ans, 3, 3);

	return 0;
}


