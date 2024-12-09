Corrected CNN Implementation

#include "hls_stream.h"
#include "ap_int.h"
#include "cnn.h"
#include <algorithm>
    using namespace std;

typedef struct
{
    DTYPE data;
    ap_uint<7> h; // 0-115
    ap_uint<7> w; // 0-115
    ap_uint<6> c; // 0-63
} DataFlow;

// 特征图分块参数
const int TILE_SIZE = 14;  // 输出特征图分块大小
const int IC_PARALLEL = 4; // 输入通道并行度
const int OC_PARALLEL = 4; // 输出通道并行度

// 读取一个输入tile的完整数据到本地buffer
void load_input_tile(
    DTYPE *input,
    DTYPE local_input[TILE_SIZE + kKernel - 1][TILE_SIZE + kKernel - 1][kNum],
    int tile_h, int tile_w)
{
#pragma HLS INLINE off

    const int in_h_start = tile_h * TILE_SIZE;
    const int in_w_start = tile_w * TILE_SIZE;
    const int in_h_end = min(in_h_start + TILE_SIZE + kKernel - 1, kInImSize);
    const int in_w_end = min(in_w_start + TILE_SIZE + kKernel - 1, kInImSize);

    // 加载整个tile区域的数据
    for (int h = in_h_start; h < in_h_end; ++h)
    {
        for (int w = in_w_start; w < in_w_end; ++w)
        {
            for (int c = 0; c < kNum; ++c)
            {
#pragma HLS PIPELINE II = 1
                int local_h = h - in_h_start;
                int local_w = w - in_w_start;
                local_input[local_h][local_w][c] =
                    input[(h * kInImSize + w) * kNum + c];
            }
        }
    }
}

// 加载一组权重到本地buffer
void load_weight_group(
    DTYPE *weight,
    DTYPE local_weight[kKernel][kKernel][OC_PARALLEL][IC_PARALLEL],
    int oc_start, int ic_start)
{
#pragma HLS INLINE off

    for (int kh = 0; kh < kKernel; ++kh)
    {
        for (int kw = 0; kw < kKernel; ++kw)
        {
            for (int oc = 0; oc < OC_PARALLEL; ++oc)
            {
                for (int ic = 0; ic < IC_PARALLEL; ++ic)
                {
#pragma HLS PIPELINE II = 1
                    if ((oc_start + oc < kNum) && (ic_start + ic < kNum))
                    {
                        local_weight[kh][kw][oc][ic] = weight[((kh * kKernel + kw) * kNum + (ic_start + ic)) * kNum +
                                                              (oc_start + oc)];
                    }
                }
            }
        }
    }
}

// 计算一个tile的卷积
void compute_tile(
    DTYPE local_input[TILE_SIZE + kKernel - 1][TILE_SIZE + kKernel - 1][kNum],
    DTYPE local_weight[kKernel][kKernel][OC_PARALLEL][IC_PARALLEL],
    DTYPE local_output[TILE_SIZE][TILE_SIZE][OC_PARALLEL],
    int oc_start, int ic_start)
{
#pragma HLS INLINE off

    // 对tile内每个输出位置计算部分结果
    for (int h = 0; h < TILE_SIZE; ++h)
    {
        for (int w = 0; w < TILE_SIZE; ++w)
        {
            // 累积一组输入通道的结果
            for (int kh = 0; kh < kKernel; ++kh)
            {
                for (int kw = 0; kw < kKernel; ++kw)
                {
                    for (int oc = 0; oc < OC_PARALLEL; ++oc)
                    {
#pragma HLS PIPELINE II = 1
                        for (int ic = 0; ic < IC_PARALLEL; ++ic)
                        {
#pragma HLS UNROLL
                            if (ic_start == 0 && kh == 0 && kw == 0)
                            {
                                local_output[h][w][oc] = 0; // 初始化
                            }
                            local_output[h][w][oc] +=
                                local_input[h + kh][w + kw][ic_start + ic] *
                                local_weight[kh][kw][oc][ic];
                        }
                    }
                }
            }
        }
    }
}

// 将计算结果写回全局内存
void write_tile_result(
    DTYPE local_output[TILE_SIZE][TILE_SIZE][OC_PARALLEL],
    DTYPE *output,
    int tile_h, int tile_w, int oc_start)
{
#pragma HLS INLINE off

    const int out_h_start = tile_h * TILE_SIZE;
    const int out_w_start = tile_w * TILE_SIZE;

    for (int h = 0; h < TILE_SIZE; ++h)
    {
        for (int w = 0; w < TILE_SIZE; ++w)
        {
            for (int oc = 0; oc < OC_PARALLEL; ++oc)
            {
#pragma HLS PIPELINE II = 1
                int gh = out_h_start + h;
                int gw = out_w_start + w;
                if (gh < kOutImSize && gw < kOutImSize && (oc_start + oc) < kNum)
                {
                    output[(gh * kOutImSize + gw) * kNum + (oc_start + oc)] =
                        local_output[h][w][oc];
                }
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

        // 本地缓存
        DTYPE local_input[TILE_SIZE + kKernel - 1][TILE_SIZE + kKernel - 1][kNum];
#pragma HLS ARRAY_PARTITION variable = local_input cyclic factor = IC_PARALLEL dim = 3

        DTYPE local_weight[kKernel][kKernel][OC_PARALLEL][IC_PARALLEL];
#pragma HLS ARRAY_PARTITION variable = local_weight complete dim = 3
#pragma HLS ARRAY_PARTITION variable = local_weight complete dim = 4

        DTYPE local_output[TILE_SIZE][TILE_SIZE][OC_PARALLEL];
#pragma HLS ARRAY_PARTITION variable = local_output complete dim = 3

        // 按tile遍历整个特征图
        for (int th = 0; th < (kOutImSize + TILE_SIZE - 1) / TILE_SIZE; ++th)
        {
            for (int tw = 0; tw < (kOutImSize + TILE_SIZE - 1) / TILE_SIZE; ++tw)
            {
                // 加载当前tile的输入数据
                load_input_tile(input, local_input, th, tw);

                // 对输出通道分组处理
                for (int oc = 0; oc < kNum; oc += OC_PARALLEL)
                {
                    // 初始化输出缓存
                    for (int h = 0; h < TILE_SIZE; ++h)
                    {
                        for (int w = 0; w < TILE_SIZE; ++w)
                        {
                            for (int i = 0; i < OC_PARALLEL; ++i)
                            {
#pragma HLS PIPELINE II = 1
                                local_output[h][w][i] = 0;
                            }
                        }
                    }

                    // 对输入通道分组累积
                    for (int ic = 0; ic < kNum; ic += IC_PARALLEL)
                    {
                        // 加载对应的权重组
                        load_weight_group(weight, local_weight, oc, ic);
                        // 计算部分结果
                        compute_tile(local_input, local_weight, local_output, oc, ic);
                    }

                    // 写回当前输出通道组的结果
                    write_tile_result(local_output, output, th, tw, oc);
                }
            }
        }
    }
}
