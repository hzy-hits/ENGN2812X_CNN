#include "hls_stream.h"
#include "ap_int.h"
#include "cnn.h"

// 定义数据流类型
typedef struct
{
    DTYPE data;
    ap_uint<16> h;
    ap_uint<16> w;
    ap_uint<16> c;
} DataFlow;

// 读取输入数据到stream
void read_input(DTYPE *input, hls::stream<DataFlow> &in_stream)
{
#pragma HLS INLINE off
    for (int h = 0; h < kInImSize; ++h)
    {
        for (int w = 0; w < kInImSize; ++w)
        {
#pragma HLS PIPELINE II = 1
            for (int c = 0; c < kNum; ++c)
            {
                DataFlow temp;
                temp.data = input[(h * kInImSize + w) * kNum + c];
                temp.h = h;
                temp.w = w;
                temp.c = c;
                in_stream.write(temp);
            }
        }
    }
}

// 读取权重到stream
void read_weight(DTYPE *weight, hls::stream<DTYPE> &weight_stream)
{
#pragma HLS INLINE off
READ_WEIGHT:
    for (int p = 0; p < kKernel; ++p)
    {
        for (int q = 0; q < kKernel; ++q)
        {
            for (int i = 0; i < kNum; ++i)
            {
                for (int j = 0; j < kNum; ++j)
                {
#pragma HLS PIPELINE II = 1
                    weight_stream.write(weight[((p * kKernel + q) * kNum + i) * kNum + j]);
                }
            }
        }
    }
}

// 计算单元 - 处理8x32的tile
void compute_unit(
    hls::stream<DataFlow> &in_stream,
    hls::stream<DTYPE> &weight_stream,
    hls::stream<DataFlow> &out_stream,
    int tile_oc_start)
{
#pragma HLS INLINE off
    const int TILE_IC = 8;
    const int TILE_OC = 32;

    // 本地缓存
    DTYPE local_input[kKernel][kKernel][TILE_IC];
#pragma HLS ARRAY_PARTITION variable = local_input complete dim = 0
    DTYPE local_weight[kKernel][kKernel][TILE_OC][TILE_IC];
#pragma HLS ARRAY_PARTITION variable = local_weight complete dim = 0
    DTYPE local_output[TILE_OC];
#pragma HLS ARRAY_PARTITION variable = local_output complete dim = 0

    // 主计算循环
    for (int h = 0; h < kOutImSize; ++h)
    {
        for (int w = 0; w < kOutImSize; ++w)
        {
            // 初始化输出缓存
            for (int oc = 0; oc < TILE_OC; ++oc)
            {
#pragma HLS UNROLL
                local_output[oc] = 0;
            }

            // 计算当前输出位置的卷积
            for (int ic_base = 0; ic_base < kNum; ic_base += TILE_IC)
            {
                // 加载输入数据
                for (int kh = 0; kh < kKernel; ++kh)
                {
                    for (int kw = 0; kw < kKernel; ++kw)
                    {
                        for (int ic = 0; ic < TILE_IC; ++ic)
                        {
#pragma HLS PIPELINE II = 1
                            DataFlow temp = in_stream.read();
                            local_input[kh][kw][ic] = temp.data;
                        }
                    }
                }

                // 加载权重
                for (int kh = 0; kh < kKernel; ++kh)
                {
                    for (int kw = 0; kw < kKernel; ++kw)
                    {
                        for (int oc = 0; oc < TILE_OC; ++oc)
                        {
                            for (int ic = 0; ic < TILE_IC; ++ic)
                            {
#pragma HLS PIPELINE II = 1
                                local_weight[kh][kw][oc][ic] = weight_stream.read();
                            }
                        }
                    }
                }

                // 计算卷积
                for (int kh = 0; kh < kKernel; ++kh)
                {
                    for (int kw = 0; kw < kKernel; ++kw)
                    {
                        for (int oc = 0; oc < TILE_OC; ++oc)
                        {
#pragma HLS PIPELINE II = 1
                            for (int ic = 0; ic < TILE_IC; ++ic)
                            {
#pragma HLS UNROLL
                                local_output[oc] += local_input[kh][kw][ic] *
                                                    local_weight[kh][kw][oc][ic];
                            }
                        }
                    }
                }
            }

            // 输出结果
            for (int oc = 0; oc < TILE_OC; ++oc)
            {
#pragma HLS PIPELINE II = 1
                DataFlow temp;
                temp.data = local_output[oc];
                temp.h = h;
                temp.w = w;
                temp.c = tile_oc_start + oc;
                out_stream.write(temp);
            }
        }
    }
}

// 写回结果
void write_output(hls::stream<DataFlow> &out_stream, DTYPE *output)
{
#pragma HLS INLINE off
    DTYPE local_output[kOutImSize][kOutImSize][kNum] = {0};
#pragma HLS RESOURCE variable = local_output core = RAM_1P_URAM

    // 收集所有计算单元的输出
    for (int h = 0; h < kOutImSize; ++h)
    {
        for (int w = 0; w < kOutImSize; ++w)
        {
            for (int c = 0; c < kNum; ++c)
            {
#pragma HLS PIPELINE II = 1
                DataFlow temp = out_stream.read();
                local_output[temp.h][temp.w][temp.c] = temp.data;
            }
        }
    }

    // 写回全局内存
    for (int h = 0; h < kOutImSize; ++h)
    {
        for (int w = 0; w < kOutImSize; ++w)
        {
#pragma HLS PIPELINE II = 1
            for (int c = 0; c < kNum; ++c)
            {
                output[(h * kOutImSize + w) * kNum + c] = local_output[h][w][c];
            }
        }
    }
}

extern "C"
{
    void cnn(DTYPE *input, DTYPE *weight, DTYPE *output)
    {
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // 定义数据流
#pragma HLS DATAFLOW
        hls::stream<DataFlow> in_stream("in_stream");
#pragma HLS STREAM variable = in_stream depth = 512
        hls::stream<DTYPE> weight_stream("weight_stream");
#pragma HLS STREAM variable = weight_stream depth = 512
        hls::stream<DataFlow> out_stream("out_stream");
#pragma HLS STREAM variable = out_stream depth = 512

        // 启动数据流处理单元
        read_input(input, in_stream);
        read_weight(weight, weight_stream);
        compute_unit(in_stream, weight_stream, out_stream, 0);
        write_output(out_stream, output);
    }
}
