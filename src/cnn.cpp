/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce
latency and
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are
used in kernel.
*******************************************************************************/

#include "hls_stream.h"
#include "ap_int.h"

#include "cnn.h"

extern "C" {

void cnn(DTYPE *input,  DTYPE *weight, DTYPE *output)
{
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    DTYPE local_input[kInImSize][kInImSize][kNum];
#pragma HLS RESOURCE variable=local_input core=RAM_1P_URAM
    DTYPE local_output[kOutImSize][kOutImSize][kNum];
#pragma HLS RESOURCE variable=local_output core=RAM_1P_URAM
    DTYPE local_weight[kKernel][kKernel][kNum][kNum];
    for (int h = 0; h < kInImSize; ++h) {
        for (int w = 0; w < kInImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_input[h][w][i] = input[(h*kInImSize+w)*kNum+i];
            }
        }
    }
    for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    local_weight[p][q][i][j] = weight[((p*kKernel+q)*kNum+i)*kNum+j];
                }
            }
        }
    }
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_output[h][w][i] = 0.0f;
            }
        }
    }
	// Convolution
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
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                output[(h*kOutImSize + w)*kNum+i] = local_output[h][w][i];
            }
        }
    }


}

}
