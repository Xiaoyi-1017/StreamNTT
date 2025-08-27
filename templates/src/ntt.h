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
using Data = ap_uint<K>; // ap_uint<14|23|32>
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

#if MOD == 12289
    #define USE_Q12289
#elif MOD == 7681
    #define USE_Q7681
#elif MOD == 8380417
    #define USE_Q8380417
#elif MOD == 3221225473
    #define USE_Q3221225473
#else
    #error "Unsupported mod value. Please define MOD as 12289 or 7681 or 8380417 or 3221225473."
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

// Bit reversed array of twiddle factors
const Data tw_factors[n] = {{TW_FACTORS}};

#endif // NTT_H
