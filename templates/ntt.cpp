#include <iostream>
#include "ntt.h"

ap_uint<logDEPTH-1> bitrev(ap_uint<logDEPTH-1> x) {
#pragma HLS INLINE
	ap_uint<logDEPTH-1> y = 0;
	for (int i = 0; i < logDEPTH-1; ++i) {
#pragma HLS UNROLL
		y[logDEPTH - 2 - i] = x[i];
	}
	return y;
}


// Non-standard DSP tiling helpers are generated in ntt_mul.h
// Helpers:
//   mul_full_data_nonstd
//   mul_low_data_data_kplus2_nonstd
//   mul_w1_compiletime
#include "ntt_mul.h"

// [reduce:begin]
// Two-class Barrett-style reduce
// Selected by USE_REDUCE_SHIFTADD when the generated shift-add reducer is used

// cls == 0: 52-class / 51/52-bit family
// cls == 1: 62-class / 61/62-bit family
#ifndef SINGLE_PRIME
void reduce(Data A, Data B, Data &Z, int mod_id){
#else
void reduce(Data A, Data B, Data &Z){
#endif
#pragma HLS INLINE

	// --- Full product (wide, single path); manually tiled ---
	Data2 U = mul_full_data_nonstd(A, B);
#ifndef SINGLE_PRIME
	Data q = MODS[mod_id];
#else
	Data q = MOD;
#endif

#if defined(USE_REDUCE_SHIFTADD)
	// BU shift-add (fold-based reducer)
#ifndef SINGLE_PRIME
	Z = reduce_shiftadd(U, (ModId)mod_id, q);
#else
	Z = reduce_shiftadd(U, (ModId)0, q);   // single-prime: only recipe index 0 exists
#endif
#elif defined(USE_REDUCE_SHIFTADD_MUL)
	// BU shift-add multiplication (Barrett/Shoup-preserving)
#ifndef SINGLE_PRIME
	Z = reduce_shiftadd_mul(U, (ModId)mod_id, q);
#else
	Z = reduce_shiftadd_mul(U, (ModId)0, q);   // single-prime: only recipe index 0 exists
#endif
#else
	// Mixed K=62 main path with explicit _cls0 candidates ONLY at recipesensitive points
#ifndef SINGLE_PRIME
	const int cls = PRIME_CLASS[mod_id];
	Data2 T = BARRETT_MUS[mod_id];
#else
	const int cls = PRIME_CLASS;
	Data2 T = BARRETT_MU;
#endif

	// V shift (recipe-sensitive)
	Dataplus V_cls0 = (Dataplus)(U >> 51);
	Dataplus V = (Dataplus)(U >> (K - 1));
	Dataplus V_mux = cls ? V : V_cls0;

	// V_l, V_h: 62-compatible (R1, R2 PASS).
	Data V_l = (Data)((ap_uint<62>)V_mux);
	ap_uint<1> V_h = (ap_uint<1>)(V_mux >> 62);

	// T_l / T_h split (recipe-sensitive)
	Data T_l_cls0 = (Data)((ap_uint<52>)T);
	Data T_l = (Data)((ap_uint<K>)T);
	Data T_l_mux = cls ? T_l : T_l_cls0;

	Data T_h_cls0 = (Data)((ap_uint<52>)(T >> 52));
	Data T_h = (Data)((ap_uint<K>)(T >> K));
	Data T_h_mux = cls ? T_h : T_h_cls0;

	// Wide multiplications (single path, operands already muxed)
	Data2 W0 = mul_full_data_nonstd(V_l, T_l_mux);
	Data2 W1 = mul_w1_compiletime(V_l, T_h_mux);
	Data2 W2 = V_h ? (Data2)T_l_mux : (Data2)0;

	// W3: 62-compatible (R3 PASS); use 62-style with T_h_mux.
	Data2 W3 = V_h ? ((Data2)T_h_mux << 61) : (Data2)0;

	// W_buffer shift/reduction (recipe-sensitive)
	Data2 W12 = (Data2)(W1 + W2);

	ap_uint<2*K+1> W_pre_cls0 =
	    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << 52);
	Data2 W_shifted_cls0 = (Data2)(W_pre_cls0 >> 53);

	ap_uint<2*K+1> W_pre =
	    (ap_uint<2*K+1>)W0 + ((ap_uint<2*K+1>)W12 << K);
	Data2 W_shifted = (Data2)(W_pre >> (K + 1));

	Data2 W_shifted_mux = cls ? W_shifted : W_shifted_cls0;
	Data2 W = W_shifted_mux + W3;

	// W_l, W_h: 62-compatible (R4, R5 PASS).
	Data W_l = (Data)((ap_uint<62>)W);
	ap_uint<1> W_h = (ap_uint<1>)(W >> 62);

	// IntMult3 (single path)
	Data2 X0 = mul_low_data_data_kplus2_nonstd(W_l, q);

	// X1: 62-compatible (R6 PASS).
	Data2 X1 = W_h ? ((Data2)q << 62) : (Data2)0;

	// mask / ring (recipe-sensitive)
	Data2 mask_cls0 = ((Data2)1 << 54) - 1;
	Data2 mask = ((Data2)1 << (K + 2)) - 1;
	Data2 mask_mux = cls ? mask : mask_cls0;

	Data2 ring_cls0 = ((Data2)1 << 54);
	Data2 ring = ((Data2)1 << (K + 2));
	Data2 ring_mux = cls ? ring : ring_cls0;

	Data2 X = (X0 + X1) & mask_mux;
	Data2 Y = U & mask_mux;

	Dataplus2 Z0 = (Dataplus2)((Y + ring_mux - X) & mask_mux);
	Dataplus2 Z1 = Z0;
	Dataplus2 two_q = (Dataplus2)q << 1;
	Dataplus2 Z2 = Z1 - (Dataplus2)q;
	Dataplus2 Z3 = Z1 - two_q;
	Dataplus2 Z_buffer = (Z1 >= two_q) ? Z3
	                  : ((Z1 >= (Dataplus2)q) ? Z2 : Z1);
	Z = static_cast<Data>(Z_buffer);
#endif  // USE_REDUCE_SHIFTADD
}
// [reduce:end]
/*
void reduce(Data coeff, Data tw_factor, Data &remainder){
#pragma HLS INLINE

	Data2 q = MOD;

        // Full 2K-bit product
        Data2 x = (Data2)coeff * (Data2)tw_factor;
    
        // BARRETT_MU = floor(2^(2K) / Q)

        // q1 = floor(x / 2^K)
        Data2 q1 = x >> K;
        // q2 = q1 * MU
        Data2 q2 = q1 * (Data2)BARRETT_MU;
        // q3 = floor(q2 / 2^K)
        Data2 q3 = q2 >> K;
        // r = x - q3 * Q
        Data2 r = x - q3 * q;
        // Correct at most twice
        if (r >= q) r -= q;
        if (r >= q) r -= q;

	remainder =  static_cast<Data>(r);
}
*/

#ifndef SINGLE_PRIME
void butterfly(Data even, Data odd, Data tw_factor, Data *out_even, Data* out_odd, int mod_id) {
#else
void butterfly(Data even, Data odd, Data tw_factor, Data *out_even, Data* out_odd) {
#endif
#pragma HLS INLINE

#ifndef SINGLE_PRIME
	ap_uint<K+2> mod = MODS[mod_id];
#else
	ap_uint<K+2> mod = MOD;
#endif

	Data reduced;
#ifndef SINGLE_PRIME
	reduce(odd, tw_factor, reduced, mod_id);
#else
	reduce(odd, tw_factor, reduced);
#endif

	ap_int<K+2> coeff_even = (ap_int<K+2>) even + (ap_int<K+2>) reduced;
	ap_int<K+2> coeff_odd  = (ap_int<K+2>) even - (ap_int<K+2>) reduced;

	ap_int<K+2> out0 = (coeff_even >= mod) ? (ap_int<K+2>) (coeff_even - mod): coeff_even;
	ap_int<K+2> out1 = (coeff_odd < 0)     ? (ap_int<K+2>) (coeff_odd + mod) : coeff_odd;

	// Cast the results to the correct width so that both branches are the same type
	*out_even = static_cast<Data>(out0);
	*out_odd  = static_cast<Data>(out1);
}

#if defined(USE_XSTAGE_S3_TW_SCALE_BU_COMPLETION) && NUM_X_STAGE == 4

// X-stage 3 BU-side tw_scale completion helper.
// Multiply the temporal base twiddle by the spatial scale.
void reduce_x_s3_tw_scale_completion(
	Data base_tw,
	Data tw_scale,
	Data &completed_tw
#ifndef SINGLE_PRIME
	, int mod_id
#endif
) {
#pragma HLS INLINE
#ifndef SINGLE_PRIME
	reduce(base_tw, tw_scale, completed_tw, mod_id);
#else
	reduce(base_tw, tw_scale, completed_tw);
#endif
}

// X-stage 3 factored butterfly: apply the spatial scale, then the butterfly.
// Multiplication by the completed twiddle remains inside butterfly().
void butterfly_x_s3_tw_scale(
	Data even,
	Data odd,
	Data base_tw,
	const int spatial_group,
	Data *out_even,
	Data *out_odd
#ifndef SINGLE_PRIME
	, int mod_id
#endif
) {
#pragma HLS INLINE
	Data completed_tw;
#ifndef SINGLE_PRIME
	reduce_x_s3_tw_scale_completion(
		base_tw,
		tw_scale_x_s3_tab[mod_id][spatial_group],
		completed_tw,
		mod_id
	);
	butterfly(even, odd, completed_tw, out_even, out_odd, mod_id);
#else
	reduce_x_s3_tw_scale_completion(
		base_tw,
		tw_scale_x_s3_tab[spatial_group],
		completed_tw
	);
	butterfly(even, odd, completed_tw, out_even, out_odd);
#endif
}

#endif // USE_XSTAGE_S3_TW_SCALE_BU_COMPLETION && NUM_X_STAGE == 4

#if defined(USE_XSTAGE_TW_SCALE_BU_COMPLETION)

// General X-stage tw_scale completion helper.
// Multiply the temporal base twiddle by the spatial scale.
void reduce_x_tw_scale_completion(
	Data base_tw,
	Data tw_scale,
	Data &completed_tw
#ifndef SINGLE_PRIME
	, int mod_id
#endif
) {
#pragma HLS INLINE
#ifndef SINGLE_PRIME
	reduce(base_tw, tw_scale, completed_tw, mod_id);
#else
	reduce(base_tw, tw_scale, completed_tw);
#endif
}

// General factored butterfly: caller provides the tw_scale value directly.
// Multiplication by the completed twiddle remains inside butterfly().
void butterfly_x_tw_scale(
	Data even,
	Data odd,
	Data base_tw,
	Data tw_scale,
	Data *out_even,
	Data *out_odd
#ifndef SINGLE_PRIME
	, int mod_id
#endif
) {
#pragma HLS INLINE
	Data completed_tw;
#ifndef SINGLE_PRIME
	reduce_x_tw_scale_completion(base_tw, tw_scale, completed_tw, mod_id);
	butterfly(even, odd, completed_tw, out_even, out_odd, mod_id);
#else
	reduce_x_tw_scale_completion(base_tw, tw_scale, completed_tw);
	butterfly(even, odd, completed_tw, out_even, out_odd);
#endif
}

#endif // USE_XSTAGE_TW_SCALE_BU_COMPLETION

void bf_unit(const int stage, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i)
{
#pragma HLS INLINE
	// delta_new(stage, BU) = 2^(half_bit) = 2^(stage)
	const int shift = stage;
	const ap_uint<logDEPTH> mask = (1<<shift) -1;

	Data twf = 0;

#ifndef SINGLE_PRIME
	// --- Cyclic mod_id: polynomial boundary is the single source of truth ---
	int cur_mod_id = 0;
#endif

// [bf_unit_storage:begin]
	// memory for entry with EVEN indices
	Data mem0[DEPTH/2]; 
	Data mem1[DEPTH/2];
#pragma HLS bind_storage variable=mem0 type=RAM_S2P impl=uram 
#pragma HLS bind_storage variable=mem1 type=RAM_S2P impl=uram 

	// memory for entry with ODD indices
	Data mem2[DEPTH/2]; 
	Data mem3[DEPTH/2];
#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=bram 
#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=bram 
// [bf_unit_storage:end]

	//memory read/write data count
	ap_uint<logDEPTH> read_idx = 0;     
	ap_uint<logDEPTH> write_idx = 0;     

	ap_uint<logDEPTH> read_limit = 0;     
	ap_uint<logDEPTH> write_limit = 0;        
// [icbu_write_limit_decl:begin]
// [icbu_write_limit_decl:end]
	bool set_read_limit = 0;
	bool set_write_limit = 0;
	bool mem_empty = true;

BF_UNIT_LOOP:
	for(;;){
#pragma HLS pipeline II = 1
#pragma HLS dependence variable=mem0 type=inter false
#pragma HLS dependence variable=mem1 type=inter false
#pragma HLS dependence variable=mem2 type=inter false
#pragma HLS dependence variable=mem3 type=inter false

		// Indexing
		ap_uint<1> read_mem_idx = (read_idx >> shift) & 1; // % 2;

		ap_uint<logDEPTH> read_upper_addr = (read_idx >> (shift+1)) << (shift);
		ap_uint<logDEPTH> read_lower_addr = (read_idx & mask);
		ap_uint<logDEPTH> raddr = read_upper_addr | read_lower_addr;

		ap_uint<1> write_mem_idx = (write_idx >> shift) & 1; // % 2;

		ap_uint<logDEPTH> write_upper_addr = (write_idx >> (shift+1)) << (shift);
		ap_uint<logDEPTH> write_lower_addr = (write_idx & mask);
		ap_uint<logDEPTH> waddr = write_upper_addr | write_lower_addr;

		// Safety checks
// [icbu_write_safe:begin]
// [icbu_write_safe:end]
		bool read_safe = read_limit != read_idx || mem_empty == true;

		if( set_write_limit == 1 ){
			mem_empty = false;
		}
		else if( set_read_limit == 1 && write_idx == read_idx ){
			mem_empty = true;
		}

		if( set_read_limit == 1 ){
			read_limit = write_idx;	
			read_limit[shift] = 0;
		}
		set_read_limit = 0;

// [icbu_write_limit_shift:begin]
// [icbu_write_limit_shift:end]
		if( set_write_limit == 1 ){
			write_limit = read_idx;	
			write_limit[shift] = 0;
		}
		set_write_limit = 0;

		if( write_safe == true ){
			Data output_data0;
			Data output_data1;

			if(write_mem_idx == 0){
				output_data0 = mem0[waddr];
				output_data1 = mem2[waddr];
			}
			else{
				output_data0 = mem1[waddr];
				output_data1 = mem3[waddr];
			}

			Data2 output_data = (output_data1, output_data0);
			output_stream.write(output_data);

			if(write_mem_idx == 1){    
				set_read_limit = 1;
			}

			write_idx++;
		}

		if( read_safe == true && !input_stream.empty() ){

			Data2 in_data = input_stream.read();
			Data in_even = in_data(K-1,0);
			Data in_odd = in_data(2*K-1,K);
			Data out_even, out_odd;

			
			Data tw_tmp = tw_i.read(); // Read in on-the-fly buffer
			twf = tw_tmp;

#ifndef SINGLE_PRIME
			butterfly(in_even, in_odd, twf, &out_even, &out_odd, cur_mod_id);
#else
			butterfly(in_even, in_odd, twf, &out_even, &out_odd);
#endif

			if(read_mem_idx == 0){
				mem0[raddr] = out_even;
				mem1[raddr] = out_odd;
			}
			else{
				mem2[raddr] = out_even;
				mem3[raddr] = out_odd;
			}

			if(read_mem_idx == 1){
				set_write_limit = 1;
			}

			read_idx++;
#ifndef SINGLE_PRIME
			// Polynomial boundary: read_idx wraps at DEPTH
			if (read_idx == 0) {
				cur_mod_id++;
				if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
			}
#endif
		}
	}
}


// [L_EARLY_TWIDDLE_S1_BEGIN]
void tw_gen_L_s1(const int stage, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
	ModId cur_mod_id = 0;
	ap_uint<1> phase = 0;
	ap_uint<logDEPTH> poly_ctr = 0;
	for(;;) {
#pragma HLS PIPELINE II=1
		Data tw_val = tw_l_base_table[1 + (int)phase][(int)cur_mod_id];
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_val);
		}
		phase++;
		poly_ctr++;
		if (poly_ctr == 0) {  // ap_uint<logDEPTH> wraps at DEPTH
			cur_mod_id++;
			if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
		}
	}
}
// [L_EARLY_TWIDDLE_S1_END]

// [bf_unit_s1:begin]
// [bf_unit_s1:end]

// [l_stage_s1:begin]
// [l_stage_s1:end]


// [L_EARLY_TWIDDLE_S0_BEGIN]
void tw_gen_L_s0(const int stage, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
	ModId cur_mod_id = 0;
	ap_uint<logDEPTH> poly_ctr = 0;
	for(;;) {
#pragma HLS PIPELINE II=1
		Data tw_val = tw_l_base_table[0][(int)cur_mod_id];
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_val);
		}
		poly_ctr++;
		if (poly_ctr == 0) {  // ap_uint<logDEPTH> wraps at DEPTH
			cur_mod_id++;
			if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
		}
	}
}
// [L_EARLY_TWIDDLE_S0_END]

// [bf_unit_s0:begin]
// [bf_unit_s0:end]

// [l_stage_s0:begin]
// [l_stage_s0:end]



// ======================================================================
// Early L-stage 2: 4-entry table-driven twiddle supply.
// ======================================================================
// [L_EARLY_TWIDDLE_S2_BEGIN]
void tw_gen_L_s2(const int stage, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
	ModId cur_mod_id = 0;
	ap_uint<2> phase = 0;
	ap_uint<logDEPTH> poly_ctr = 0;
	for(;;) {
#pragma HLS PIPELINE II=1
		Data tw_val = tw_l_base_table[3 + (int)phase][(int)cur_mod_id];
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_val);
		}
		phase++;
		poly_ctr++;
		if (poly_ctr == 0) {  // ap_uint<logDEPTH> wraps at DEPTH
			cur_mod_id++;
			if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
		}
	}
}
// [L_EARLY_TWIDDLE_S2_END]

// [bf_unit_s2:begin]
// [bf_unit_s2:end]

// [l_stage_s2:begin]
// [l_stage_s2:end]

// ======================================================================
// Early L-stage 3: 8-entry table-driven twiddle supply.
// ======================================================================
// [bf_unit_ge3:begin]
// [bf_unit_ge3:end]

// [L_EARLY_TWIDDLE_S3_BEGIN]
void tw_gen_L_s3(const int stage, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off
	ModId cur_mod_id = 0;
	ap_uint<3> phase = 0;
	ap_uint<logDEPTH> poly_ctr = 0;
	for(;;) {
#pragma HLS PIPELINE II=1
		Data tw_val = tw_l_base_table[7 + (int)phase][(int)cur_mod_id];
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_val);
		}
		phase++;
		poly_ctr++;
		if (poly_ctr == 0) {  // ap_uint<logDEPTH> wraps at DEPTH
			cur_mod_id++;
			if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
		}
	}
}
// [L_EARLY_TWIDDLE_S3_END]

// [l_stage_s3:begin]
// [l_stage_s3:end]

// ======================================================================
// Later L-stages: feedback base recurrence plus temporal reconstruction.
// ======================================================================
// [bf_unit_ge4:begin]
// [bf_unit_ge4:end]

// [tw_gen_L_base_s_ge4:begin]
// [tw_gen_L_base_s_ge4:end]

// [tw_complete_L_ratio_s_ge4:begin]
// [tw_complete_L_ratio_s_ge4:end]

// [l_stage_s_ge4:begin]
// [l_stage_s_ge4:end]



void x_stages(tapa::istreams<Data2, BU>& input_streams, tapa::ostreams<Data2, BU>& output_streams, tapa::istreams<Wide, num_x_stage>& tw_X_W){

	const int num_of_stages = logBU + 1;
	Data mem[num_of_stages+1][WIDTH];
	ap_uint<logDEPTH> i = 0;

#ifndef SINGLE_PRIME
	int cur_mod_id = 0;
#endif

	Data tw_local[num_x_stage][BU];
#pragma HLS ARRAY_PARTITION variable=tw_local dim=1 complete
#pragma HLS ARRAY_PARTITION variable=tw_local dim=2 complete

NTT_SPATIAL_LOOP:
	for(;;){
#pragma HLS PIPELINE II = 1

		bool do_read = 1;
		for(int j = 0; j < BU; j++){
#pragma HLS UNROLL
			do_read &= !input_streams[j].empty() ;
		}

		//for(int i = 0; i < DEPTH; i++){
		if( do_read == true ){
INPUT_LOOP:
			for(int j = 0; j < BU; j++){
#pragma HLS UNROLL
				Data2 in_data = input_streams[j].read();

				mem[0][2*j] = in_data(K-1,0);
				mem[0][2*j+1] = in_data(2*K-1,K);
			}

STAGE_LOOP:
			for(int s = 0; s < num_of_stages; s++){
#pragma HLS UNROLL 

				int current_stage = s + logN - logBU -1;
				int stage_shift = current_stage + 1;
				int stride = n >> ((current_stage) + 1);
				int next_stride = stride >> 1;

				// For division
				int shift = num_of_stages - (s + 1);

				// For remainder
				int mask = (1 << shift) -1;
				int next_mask = ( 1 << (shift-1) ) -1;
				
				Wide w = tw_X_W[s].read();
DISTRIBUTION_LOOP:
				for(int idx = 0; idx < BU; idx++){
#pragma HLS UNROLL
					//const int tw_base_idx = idx + s * BU;
					//tw_local[s][idx] = tw_X[tw_base_idx].read();
					tw_local[s][idx] = w.range((idx+1)*K-1, idx*K);
				}

// [X_STAGE_BUTTERFLY_DISPATCH_BEGIN]
BUTTERFLY_LOOP:
				for(int idx = 0; idx < BU; idx++){
#pragma HLS UNROLL
					Data out_even, out_odd;

					int j = idx >> shift;
					int k = idx & mask;

					int ind_even = (next_stride == 0) ? ( j << (shift+1) ) 
						: ( j << (shift+1) ) + (k & next_mask) * 2 + (k >> (shift-1));
					int ind_odd = ind_even + stride;
					
					Data twf = tw_local[s][idx];


#if defined(USE_XSTAGE_S3_TW_SCALE_BU_COMPLETION) && NUM_X_STAGE == 4
					if (s == num_of_stages - 1) {
						// Complete the X-stage 3 twiddle before applying the butterfly.
						// idx is the spatial group at this stage (shift=0, group=idx).
						butterfly_x_s3_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							idx,
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#endif
#if defined(USE_XSTAGE_TW_SCALE_BU_COMPLETION)
					if (s == 1) {
						// S1 BU-side completion. spatial_group = idx >> (logBU - 1).
						const int spatial_group_s1 = idx >> (logBU - 1);
						butterfly_x_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							tw_scale_x_s1_tab[cur_mod_id][spatial_group_s1],
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#if NUM_X_STAGE > 2
					if (s == 2) {
						// S2 BU-side completion. spatial_group = idx >> (logBU - 2).
						const int spatial_group_s2 = idx >> (logBU - 2);
						butterfly_x_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							tw_scale_x_s2_tab[cur_mod_id][spatial_group_s2],
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#endif
#if NUM_X_STAGE > 3
					if (s == 3) {
						// S3 BU-side completion. spatial_group = idx >> (logBU - 3).
						const int spatial_group_s3 = idx >> (logBU - 3);
						butterfly_x_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							tw_scale_x_s3_tab[cur_mod_id][spatial_group_s3],
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#if NUM_X_STAGE > 4
					if (s == 4) {
						// S4 BU-side completion. spatial_group = idx >> (logBU - 4).
						const int spatial_group_s4 = idx >> (logBU - 4);
						butterfly_x_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							tw_scale_x_s4_tab[cur_mod_id][spatial_group_s4],
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#if NUM_X_STAGE > 5
					// X-stage 5 BU-side completion.
					// spatial_group = idx >> (logBU - 5) = idx for BU=32 (shift=0).
					if (s == 5) {
						const int spatial_group_s5 = idx >> (logBU - 5);
						butterfly_x_tw_scale(
							mem[s][2*idx],
							mem[s][2*idx+1],
							twf,
							tw_scale_x_s5_tab[cur_mod_id][spatial_group_s5],
							&out_even,
							&out_odd,
							cur_mod_id
						);
					} else
#endif
#endif
#endif
#endif // USE_XSTAGE_TW_SCALE_BU_COMPLETION
					{
						butterfly(mem[s][2*idx], mem[s][2*idx+1], twf, &out_even, &out_odd, cur_mod_id);
					}
					mem[s+1][ind_even] = out_even;
					mem[s+1][ind_odd] = out_odd;

				}
// [X_STAGE_BUTTERFLY_DISPATCH_END]

			}

OUTPUT_LOOP:
			for(int j = 0; j < BU; j++){
#pragma HLS UNROLL
				Data2 out_data = (mem[num_of_stages][2*j+1], mem[num_of_stages][2*j]);
				output_streams[j].write(out_data);
			}

			i++;
#ifndef SINGLE_PRIME
			// Polynomial boundary
			if (i == 0) {
				cur_mod_id++;
				if (cur_mod_id >= NUM_PRIMES) cur_mod_id = 0;
			}
#endif
		}
	}
}


// [X_STAGE_FUNCTIONS_HERE]

// [X_STAGE_TOP_BEGIN]
// Template self-compilation stub ONLY. The segmented generator ALWAYS replaces this entire
// BEGIN..END span (re.sub in generate_seg_case) with the per-stage segmented x_stages_top
// wiring, so nothing between these markers ever appears in generated output. Kept minimal and
// self-compiling: ntt_core() references x_stages_top, so the symbol must exist for the template
// to compile standalone. Per-stage segmented TFG modules own X-stage twiddle supply, so this
// template stub intentionally has no fallback producer chain.
void x_stages_top(tapa::istreams<Data2, BU>& input_streams, tapa::ostreams<Data2, BU>& output_streams){
	(void)input_streams;
	(void)output_streams;
}
// [X_STAGE_TOP_END]

void input_mem_stage(tapa::istream<Data2>& i_stream, tapa::ostream<Data2>& o_stream){

	// memory for entry with EVEN indices
	Data mem0[2][DEPTH/2]; 
	Data mem1[2][DEPTH/2];
#pragma HLS bind_storage variable=mem0 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem1 type=RAM_S2P impl=lutram 

	// memory for entry with ODD indices
	Data mem2[2][DEPTH/2]; 
	Data mem3[2][DEPTH/2];
#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=lutram 

	//double buffer index
	ap_uint<1> rd_s = 0;
	ap_uint<1> wr_s = 1;

	//memory read/write data count
	ap_uint<logDEPTH> read_idx = 0;     
	ap_uint<logDEPTH> write_idx = 0;     

	bool read_exist = false;
	bool read_done = false;
	bool write_done = false;

INPUT_MEM_STAGE_LOOP:
	for(;;){
#pragma HLS pipeline II = 1
#pragma HLS dependence variable=mem0 type=inter false
#pragma HLS dependence variable=mem1 type=inter false
#pragma HLS dependence variable=mem2 type=inter false
#pragma HLS dependence variable=mem3 type=inter false

		if( read_exist == true && write_done == false ){
			Data data_e;
			Data data_o;
			
			// Write out bitrev(addr)
			ap_uint<logDEPTH-1> addr = (ap_uint<logDEPTH-1>)(write_idx);
			ap_uint<logDEPTH-1> addr_perm = bitrev(addr);
			

			// For even inputs 0 ~ n/2-1
			if (write_idx < DEPTH/2){ // 0, 512 - 2, 514 ...
				data_e = mem0[wr_s][addr_perm];
				data_o = mem2[wr_s][addr_perm];
			}
			else{            // 1, 513 - 3, 515 ...
				data_e = mem1[wr_s][addr_perm];
				data_o = mem3[wr_s][addr_perm];
			}

			Data2 data = (data_o, data_e);
			o_stream.write(data);

			write_idx++;
			if(write_idx == 0){
				write_done = true;
			} 
		}

		if( read_done == false && !i_stream.empty() ){
			Data2 data = i_stream.read();
			Data data_e = data(K-1,0);
			Data data_o = data(2*K-1,K);

			if(read_idx < DEPTH/2){
				// For even inputs 0 ~ n/2-1      
				mem0[rd_s][read_idx] = data_e;
				mem1[rd_s][read_idx] = data_o;
			}
			else{
				// For odd inputs n/2 ~ n-1
				mem2[rd_s][read_idx-DEPTH/2] = data_e;
				mem3[rd_s][read_idx-DEPTH/2] = data_o;
			}

			read_idx++;
			if(read_idx == 0){
				read_done = true;
			} 
		}

		//Go to the next polynomial when all existing data has been written out
		//and when either reading data has finished or no data has been read
		if( read_idx == 0 && (read_exist == false || write_done == true) ){
			if(read_done == true){
				read_exist = true;
			}
			else{
				read_exist = false;
			}
			read_done = false;
			write_done = false;

			wr_s = rd_s;
			rd_s ^= (ap_uint<1>)1;
		}
	}
}

#if HOST_PREPERMUTE_INPUT
void input_stage(tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream){
#pragma HLS INLINE off
	for(;;){
#pragma HLS PIPELINE II = 1
		output_stream.write(input_stream.read());
	}
}
#endif


void ntt_core(tapa::istreams<Data2, BU> core_istreams, tapa::ostreams<Data2, BU> core_ostreams){

	tapa::streams<Data2, BU*(num_l_stage+1), 2> core_streams("core_streams");

	// [NTT_CORE_LSTAGE_CHAIN_BEGIN]
	tapa::task()
#if HOST_PREPERMUTE_INPUT
		.invoke<tapa::detach, BU>(input_stage, core_istreams, core_streams)
#else
		.invoke<tapa::detach, BU>(input_mem_stage, core_istreams, core_streams)
#endif
		.invoke<tapa::detach>(l_stage_s0, 0, core_streams, core_streams)
		.invoke<tapa::detach>(l_stage_s1, 1, core_streams, core_streams)
#if NUM_L_stage > 2
		.invoke<tapa::detach>(l_stage_s2, 2, core_streams, core_streams)
#endif
#if NUM_L_stage > 3
		.invoke<tapa::detach>(l_stage_s3, 3, core_streams, core_streams)
#endif
#if NUM_L_STAGE_GE4 > 0
		.invoke<tapa::detach, NUM_L_STAGE_GE4>(l_stage_s_ge4, tapa::seq(), core_streams, core_streams)
#endif
		.invoke<tapa::detach>(x_stages_top, core_streams, core_ostreams);
	// [NTT_CORE_LSTAGE_CHAIN_END]
}

// [top_dram_ntt:begin]
// [top_dram_ntt:end]
