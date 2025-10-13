#ifndef NTT_H
#define NTT_H

#include <cmath>
#include <algorithm>
#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <tapa.h>

constexpr int K = {K};

using HostData = {DATA_FORMAT}; 
using Data = ap_uint<K>;
using Data2 = ap_uint<K*2>;
 
constexpr int DataCHLen = 64 / {DATA_BSIZE}; // 64 / sizeof(Data)
constexpr int EffDataCHLen = {EffDataCHLen}; 

using DataVec = tapa::vec_t<HostData, DataCHLen>;

template <typename T>
using bits = ap_uint<tapa::widthof<T>()>;

constexpr int log2(int x) {
    return (x <= 1) ? 0 : 1 + log2(x / 2);
}

#define MOD {MOD}

#if MOD == 3221225473
constexpr unsigned long long BARRETT_MU = 5726623059;
#else
constexpr unsigned long BARRETT_MU = (1ULL << (2 * K)) / MOD;
#endif

// Number of coefficients
constexpr int n = {N};
constexpr int logN = {logN};

constexpr int BU = {BU};
constexpr int logBU = {logBU};

// WIDTH: number of coeffs processed in parallel (per NTT CORE)
constexpr int WIDTH = 2*BU;
constexpr int DEPTH = n / WIDTH;
constexpr int logDEPTH = {logDEPTH};

constexpr int num_x_stage = logBU + 1;
constexpr int num_l_stage = logN - (logBU + 1);

constexpr int CH = {CH};

constexpr int NUM_CORE = EffDataCHLen*CH / (2*BU);

constexpr int GROUP_NUM = {GROUP_NUM}; //std::min(CH, NUM_CORE);
#define GROUP_CH_NUM {GROUP_CH_NUM}
//constexpr int GROUP_CH_NUM = CH / GROUP_NUM;
constexpr int GROUP_CORE_NUM = NUM_CORE / GROUP_NUM;

#if {MCH} // GROUP_CH_NUM > 1
#define MCH
#endif

constexpr int DRAM_CORE_WIDTH_RATIO = DataCHLen / (2*BU);
constexpr int CORE_DRAM_WIDTH_RATIO = (2*BU) / DataCHLen;

constexpr int POLY_FIFO_DEPTH_S = n / DataCHLen;
constexpr int POLY_FIFO_DEPTH_M = n / (2*BU);

constexpr HostData psi = {PSI};

// Small pre-computed basic twiddle factor vectors 
constexpr int tw_x_base_size = 2*BU-1;
{TWF_L_BASE}
{TWF_X_BASE}

using Wide = ap_uint<BU * K>;

#endif // NTT_H
