/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <iostream>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <math.h>
#include "cnn.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

float IsError(float a, float b) {
	return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

void cnn_sw(std::vector<DTYPE> input, std::vector<DTYPE> weight, std::vector<DTYPE> & output){

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if( tid == 0 ){
            int nthreads = omp_get_num_threads();
            std::cout << "Running OpenMP with " << nthreads << " threads...\n";
        }
    }

	// Allocate memory on heap to avoid stack overflow.
    static float local_input[kInImSize][kInImSize][kNum];
    static float local_output[kOutImSize][kOutImSize][kNum];
    static float local_weight[kKernel][kKernel][kNum][kNum];
#pragma omp parallel for
    for (int h = 0; h < kInImSize; ++h) {
        for (int w = 0; w < kInImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_input[h][w][i] = input[(h*kInImSize+w)*kNum+i];
            }
        }
    }
#pragma omp parallel for
    for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    local_weight[p][q][i][j] = weight[((p*kKernel+q)*kNum+i)*kNum+j];
                }
            }
        }
    }
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_output[h][w][i] = 0.0f;
            }
        }
    }
	// Convolution
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w) {
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    for (int p = 0; p < kKernel; ++p) {
                        for (int q = 0; q < kKernel; ++q){
                            local_output[h][w][i] += local_input[h+p][w+q][j] * local_weight[p][q][i][j];
                        }
                    }
				}
			}
		}
	}
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                output[(h*kOutImSize + w)*kNum+i] = local_output[h][w][i];
            }
        }
    }
}

const int input_size  = kNum*kInImSize*kInImSize;
const int weight_size = kNum*kNum*kKernel*kKernel;
const int output_size = kNum*kOutImSize*kOutImSize;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    
    //////////////////////////////////////////
    // Open xclbin
    //////////////////////////////////////////
    auto device = xrt::device(0); //device index=0
    char* xclbinFilename=argv[1];
    std::cout << "Open the device " << 0 << std::endl;
    std::cout << "Load the xclbin " << xclbinFilename << std::endl;
	auto uuid = device.load_xclbin(xclbinFilename);
	auto dhdl = xrtDeviceOpenFromXcl(device);
    size_t input_size_bytes  = sizeof(DTYPE) * input_size ;
    size_t weight_size_bytes = sizeof(DTYPE) * weight_size;
    size_t output_size_bytes = sizeof(DTYPE) * output_size;
    auto krnl = xrt::kernel(device, uuid, "cnn");

    // Allocate host side memory
    std::vector<DTYPE> A(input_size,1);
    std::vector<DTYPE> B(weight_size,1);
    std::vector<DTYPE> AB_sw(output_size,1);
    // Initialize the test data
    for (int i = 0; i < input_size; ++i) {
        A[i] = -2 + static_cast <float>(rand()) /( static_cast <float> (RAND_MAX/(4)));
    }
    for (int i = 0; i < weight_size; ++i) {
        B[i] = -2 + static_cast <float>(rand()) /( static_cast <float> (RAND_MAX/(4)));
    }

    //Allocate Buffer in Global Memory
    auto bo0 = xrt::bo(device, input_size_bytes, krnl.group_id(1));
    auto bo1 = xrt::bo(device, weight_size_bytes, krnl.group_id(1));
    auto bo_out = xrt::bo(device, output_size_bytes, krnl.group_id(1));

    // Map the contents of the buffer object into host memory
    auto bo0_map = bo0.map<DTYPE*>();
    auto bo1_map = bo1.map<DTYPE*>();
    auto bo_out_map = bo_out.map<DTYPE*>();

    // Create the test data
    for (int i = 0; i < input_size; ++i) {
        bo0_map[i] = A[i];
    }
    for (int i = 0; i < weight_size; ++i) {
        bo1_map[i] = B[i];
    }

    // Synchronize buffer content with device side
    bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_size_bytes, 0);
    bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE, weight_size_bytes, 0);

    std::cout << "Running FPGA MM...\n";
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();

    //Execution of the kernel
    auto run = krnl(bo0, bo1, bo_out);
    run.wait();

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::cout << "Done.\n";
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    std::cout << "Execution time = " << kernel_time_in_sec << std::endl;
    double gops = double(kKernel) * kKernel * kNum * kNum * kOutImSize * kOutImSize * 2 * 1e-9 / (kernel_time_in_sec);
    std::cout << "FPGA CNN Time: " << kernel_time_in_sec << " sec, FPGA GOPS: " << gops << std::endl;

    // Get the output data from the device;
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, output_size_bytes, 0);
    
    // Calculate the golden results
    std::cout << "Running SW CNN...\n";
    auto start = std::chrono::steady_clock::now();
    cnn_sw(A, B, AB_sw);
    auto end = std::chrono::steady_clock::now();
    double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double cpu_gops = double(kKernel) * kKernel * kNum * kNum * kOutImSize * kOutImSize * 2 / (exec_time);
    std::cout << "CPU CNN Time: " << exec_time*1e-9 << " sec,CPU CNN GOPS: " << cpu_gops << std::endl;
    printf("Done\n");

    // Validate our results
    int error = 0;
	bool first = true;
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w) {
            for (int i = 0; i < kNum; ++i) {
				if (IsError(bo_out_map[(i*kOutImSize+h)*kOutImSize+w], AB_sw[(i*kOutImSize+h)*kOutImSize+w])) {
					if (first) {
						std::cout << "First error: Got " << bo_out_map[(i*kOutImSize+h)*kOutImSize+w] << ", expecting "
							<< AB_sw[(i*kOutImSize+h)*kOutImSize+w] << " @ i = " << i << ", h = " << h
							<< ", w = " << w << std::endl;
						first = false;
					}
					++error;
				}
			}
		}
	}

    if (error != 0) {
		std::cout << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
		std::cout << "FPGA CNN FAIL" << std::endl;
		return EXIT_FAILURE;
	} else {
		std::cout << "FPGA CNN PASS" << std::endl;
		return EXIT_SUCCESS;
	}

    return 0;
}