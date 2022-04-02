#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
using namespace sycl;
#define N 1024
#define G 16
#undef NDEBUG
#include <cassert>

float base_mm(sycl::queue queue, std::vector<float>& a, std::vector<float>& b, \
	std::vector<float>& c, uint32_t size) {

	sycl::buffer<float, 1> buf_a(a.data(), a.size());
	sycl::buffer<float, 1> buf_b(b.data(), b.size());
	sycl::buffer<float, 1> buf_c(c.data(), c.size());
	sycl::event event = queue.submit([&](handler& cgh) {
		auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
		auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
		auto in_c = buf_c.get_access<sycl::access::mode::write>(cgh);

		cgh.parallel_for<class Mult>(nd_range<2>(range<2>(N, N), range<2>(G, G)), [=](nd_item<2> item) {
			size_t i = item.get_global_id(0);
			size_t j = item.get_global_id(1);
			float sum = 0.0f;
			for (size_t k = 0; k < N; ++k) {
				sum += in_a[i * N + k] * in_b[k * N + j];
			}
			in_c[i * N + j] = sum;
		});
	});
	event.wait();

	uint64_t start = event.get_profiling_info<info::event_profiling::command_start>();
	uint64_t end = event.get_profiling_info<info::event_profiling::command_end>();
	return (end - start) / 1e9;
}

float block_mm(sycl::queue q, std::vector<float>& a, std::vector<float>& b, std::vector<float>& c) {
	sycl::buffer<float, 1> buf_a(a.data(), a.size());
	sycl::buffer<float, 1> buf_b(b.data(), b.size());
	sycl::buffer<float, 1> buf_c(c.data(), c.size());

	sycl::event event = q.submit([&](handler& cgh) {
		auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
		auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
		auto in_c = buf_c.get_access<sycl::access::mode::write>(cgh);

		cgh.parallel_for<class Mult_block>(nd_range<2>(range<2>(N, N), range<2>(G, G)), [=](nd_item<2> item) {
				
			uint32_t gi = item.get_global_id(0), gj = item.get_global_id(1);
			uint32_t block_count = N / G;

			float sum = 0.0f;
			for (size_t i = 0; i < block_count; ++i) {
				for (int k = 0; k < G; ++k) {
					sum += in_a[gi * N + G * i + k] * in_b[(G * i + k) * N + gj];
				}
			}
			in_c[gi * N + gj] = sum;
		});
	});
	event.wait();

	uint64_t start = event.get_profiling_info<info::event_profiling::command_start>();
	uint64_t end = event.get_profiling_info<info::event_profiling::command_end>();
	return (end - start) / 1e9;
}

float _block_mm(sycl::queue queue, std::vector<float>& a, std::vector<float>& b, \
	std::vector<float>& c, uint32_t size, uint32_t block) {

	sycl::buffer<float, 1> buf_c(c.data(), c.size());
	sycl::buffer<float, 1> buf_a(a.data(), a.size());
	sycl::buffer<float, 1> buf_b(b.data(), b.size());

	sycl::event event = queue.submit([&](handler& cgh) {

		sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> \
			buffer(2 * block * block, cgh);

		auto in_c = buf_c.get_access<sycl::access::mode::write>(cgh);
		auto a = buf_a.get_access<sycl::access::mode::read>(cgh);
		auto b = buf_b.get_access<sycl::access::mode::read>(cgh);

		cgh.parallel_for<class _Mult>(nd_range<2>(range<2>(size, size), range<2>(block, block)), [=](nd_item<2> item) {
			float* block_a = buffer.get_pointer();
			float* block_b = block_a + block * block;

			size_t li = item.get_local_id(0);	//локальный индекс в группе (строка)
			size_t lj = item.get_local_id(1);
			uint32_t gi = block * item.get_group(0) + li;	//начало номера группы по строке 
			uint32_t gj = block * item.get_group(1) + lj;
			uint32_t block_count = size / block;

			float sum = 0.0f;
			for (size_t i = 0; i < block_count; ++i) {
				block_a[li * block + lj] = a[gi * size + block * i + lj];
				block_b[li * block + lj] = b[(block * i + li) * size + gj];
				item.barrier(sycl::access::fence_space::local_space);
				for (int k = 0; k < block; ++k) {
					sum += block_a[li * block + k] * block_b[k * block + lj];
				}
			}
			in_c[gi * size + gj] = sum;

		});
	});
	event.wait();

	uint64_t start = event.get_profiling_info<info::event_profiling::command_start>();
	uint64_t end = event.get_profiling_info<info::event_profiling::command_end>();
	return (end - start) / 1e9;
}


int main() {
	std::vector<float> a(N * N, 1.0f), b(N * N, 2.0f), c(N * N, 0.0f);
	for (int i = 0; i < 3; ++i) {
		sycl::property_list props{ sycl::property::queue::enable_profiling() };
		sycl::queue cpu_queue(sycl::cpu_selector{}, props);

		std::cout << base_mm(cpu_queue, a, b, c, N) << "\n";
		std::cout << block_mm(cpu_queue, a, b, c) << "\n";
		std::cout << _block_mm(cpu_queue, a, b, c, N, G) << "\n";
	}

	for (size_t i = 0; i < N * N; ++i)
		assert(c[i] == 2.0f * N);

	return 0;
}