#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <random>

using type = int;

type* matrix_multiplication(type* matr1, size_t n1, size_t m1, type* matr2, size_t n2, size_t m2, size_t block_sz) {
	if (m1 != n2) {
		std::cout << "Incorrect multiplication" << std::endl;
		return nullptr;
	}

	type* res_matr = new type[n1 * m2];
	//memset(res_matr, 0, sizeof(type)*n1*m2);
	for (size_t i = 0; i < n1; i++)
		for (size_t j = 0; j < m2; j++)
			res_matr[i * m2 + j] = 0;

	//#pragma omp parallel for //shared(matr1, matr2, res_matr) reduction (+:temp)
	for (size_t ib = 0; ib < n1; ib += block_sz)
		for (size_t kb = 0; kb < m1; kb += block_sz)
			for (size_t jb = 0; jb < m2; jb += block_sz)

				for (size_t i = 0; i < block_sz; ++i)
					for (size_t k = 0; k < block_sz; ++k)
						for (size_t j = 0; j < block_sz; ++j)
							res_matr[(ib + i) * m2 + (jb + j)] += \
							matr1[(ib + i) * m1 + (kb + k)] * \
							matr2[(kb + k) * m2 + (jb + j)];



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

void destructor(type* matrix) {
	if (matrix == nullptr) return;
	delete[]matrix;
	matrix = nullptr;
}

int main()
{
	srand(time(NULL));
	//..........................................................
	type* matr1 = generate_matrix(4, 4);
	type* matr2 = generate_matrix(4, 4);
	type* ans = matrix_multiplication(matr1, 4, 4, matr2, 4, 4, 2);
	for (size_t i = 0; i < 4; ++i) {
		for (size_t j = 0; j < 4; ++j)
			std::cout << ans[i * 4 + j] << ' ';
		std::cout << std::endl;
	}


	destructor(matr1);
	destructor(matr2);
	destructor(ans);

	return 0;
}


