#ifndef NTT_H
#define NTT_H

#include <cmath>
#include <algorithm>
#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <tapa.h>
#include <cinttypes>

constexpr int K = {K};

using HostData = {DATA_FORMAT}; 
using Data = ap_uint<K>;
using Data2 = ap_uint<K*2>;
//Extesion
using Dataplus = ap_uint<K+1>;
using Dataplus2 = ap_uint<K+2>;
using Dataplus3 = ap_uint<K+3>;
using Data2plus = ap_uint<2*K+1>;
 
constexpr int DataCHLen = 64 / {DATA_BSIZE}; // 64 / sizeof(Data)
constexpr int EffDataCHLen = {EffDataCHLen}; 

using DataVec = tapa::vec_t<HostData, DataCHLen>;

template <typename T>
using bits = ap_uint<tapa::widthof<T>()>;

constexpr int log2(int x) {
    return (x <= 1) ? 0 : 1 + log2(x / 2);
}

#define MOD {MOD}

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
#define NUM_L_stage {NUM_L_stage}

#define TFG_II {TFG_II}

constexpr int num_l_stage_ge1 = num_l_stage - 1;
constexpr int num_l_stage_ge2 = num_l_stage - 2;
#if TFG_II == 3 || TFG_II == 4
constexpr int num_l_stage_ge3 = num_l_stage - 3;
#endif

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
{REDUCE_BLOCK}

// Small pre-computed basic twiddle factor vectors 
constexpr int tw_x_base_size = 2*BU-1;
{DELTA}
// {TWF_L_BASE}
{TWF_L_BASE_LANE0}
{TWF_L_BASE_LANE1}
#if TFG_II == 3 || TFG_II == 4
{TWF_L_BASE_LANE2}
{TWF_L_BASE_LANE3}
#endif
{TWF_X_BASE_LANE0}
#if TFG_II != 1
{TWF_X_BASE_LANE1}
#endif
#if TFG_II == 3 || TFG_II == 4
{TWF_X_BASE_LANE2}
#endif
#if TFG_II == 4
{TWF_X_BASE_LANE3}
#endif
{TWF_L_GAMMA}
{TWF_X_GAMMA}

using Wide = ap_uint<BU * K>;
#define Match_Delay

#endif // NTT_H
