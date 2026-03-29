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

void reduce(Data A, Data B, Data &Z){
#pragma HLS INLINE
	Data q = MOD;
	Data2 T = BARRETT_MU;
	Data2 mask = ((Data2)1 << (K+2)) - 1; // keep K+2 bits: r can reach [0, 3q)

	// IntMult1: Full-IntMult, n-bits * n-bits => 2n-bits
	Data2 U = (Data2)(A * B); // 2n-bits
	ap_uint<K+1> V = static_cast<ap_uint<K+1>>(U >> (K-1)); // (n+1)-bits
	Data V_l = static_cast<Data>(V); // n-bits
	ap_uint<1> V_h = V[K]; // 1-bit

	// IntMult2: Upper half (UH)-IntMult, n-bits * n-bits => upper n-bits
	Data T_l = (Data)T;
	Data T_h = (Data)(T >> K);
	Data2 W0 = (Data2)(V_l * T_l);
	Data2 W1 = (Data2)(V_l * T_h);
	Data2 W2 = (V_h)?(Data2)T_l:(Data2)0;
	ap_uint<K+1> W3 = V_h ? ((ap_uint<K+1>)T_h << (K-1)) : (ap_uint<K+1>)0; // high_term
	ap_uint<2*K+1> W_buffer = (ap_uint<2*K+1>)( W0 + ((ap_uint<2*K+1>)(W1 + W2) << K)); // (2n+1)-bits
	ap_uint<K+1> W = static_cast<ap_uint<K+1>>(W_buffer >> (K+1)) + W3; // (n+1)-bits
	Data W_l = static_cast<Data>(W); // n-bits
	ap_uint<1> W_h = (ap_uint<1>)(W >> K);
	
	// IntMult3: Lower half (LH)-IntMult, n-bits * n-bits => lower K+2 bits
	Data2 X0 = (Data2)(W_l * q); // keep low K+2 bits
	Data2 X1 = (W_h)?(((Data2)q << K)):(Data2)0; // keep low K+2 bits
	Data2 X = (X0 + X1) & mask;
	
	Data2 Y = U & mask;
	
	// Z0 = Y - X in ring 2^(K+2); then subtract q at most twice.
	ap_uint<K+2> Z0 = (ap_uint<K+2>)((Y +(((Data2)1)<<(K+2))-X) & mask);
	Dataplus2 Z1 = static_cast<Dataplus2>(Z0);
	Dataplus2 two_q = (Dataplus2)q << 1;
	Dataplus2 Z2 = Z1 - (Dataplus2)q;
	Dataplus2 Z3 = Z1 - two_q;
	Dataplus2 Z_buffer = (Z1>=two_q)?Z3:((Z1>=(Dataplus2)q)?Z2:Z1);
	Z = static_cast<Data>(Z_buffer);
}
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

void butterfly(Data even, Data odd, Data tw_factor, Data *out_even, Data* out_odd) {
#pragma HLS INLINE

	ap_uint<K+2> mod = MOD;

	Data reduced;
	reduce(odd, tw_factor, reduced);

	ap_int<K+2> coeff_even = (ap_int<K+2>) even + (ap_int<K+2>) reduced;
	ap_int<K+2> coeff_odd  = (ap_int<K+2>) even - (ap_int<K+2>) reduced;

	ap_int<K+2> out0 = (coeff_even >= mod) ? (ap_int<K+2>) (coeff_even - mod): coeff_even;
	ap_int<K+2> out1 = (coeff_odd < 0)     ? (ap_int<K+2>) (coeff_odd + mod) : coeff_odd;

	// Cast the results to the correct width so that both branches are the same type
	*out_even = static_cast<Data>(out0);
	*out_odd  = static_cast<Data>(out1);
}

void bf_unit(const int stage, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i)
{
#pragma HLS INLINE	
	// delta_new(stage, BU) = 2^(half_bit) = 2^(stage)
	const int shift = stage;
	const ap_uint<logDEPTH> mask = (1<<shift) -1;
	
	Data twf = 0;
	// const Data L_BASE_s =tw_l_base[stage]; 
	// const Data R_s = (stage)? tw_l_base[stage-1]: (Data)1;

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

	//memory read/write data count
	ap_uint<logDEPTH> read_idx = 0;     
	ap_uint<logDEPTH> write_idx = 0;     

	ap_uint<logDEPTH> read_limit = 0;     
	ap_uint<logDEPTH> write_limit = 0;        
	ap_uint<logDEPTH> write_limit_1d = 0;  // write-in with a one-cycle delay
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
#ifdef Match_Delay // Match the delay of different RAMs r&w (e.g., 4-URAM)
		bool write_safe = write_limit_1d != write_idx;
#else
		bool write_safe = write_limit != write_idx;
#endif
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

#ifdef Match_Delay
		write_limit_1d = write_limit;
#endif
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

			butterfly(in_even, in_odd, twf, &out_even, &out_odd);

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
		}
	}
}

void reduce_tw(Data A, Data B, Data gamma,  Data &Z){
#pragma HLS INLINE
	const Data q = (Data)MOD;
	
	// X = A * B
	Dataplus X = (Dataplus)((Data2)A * (Data2)B);
	
	// t = A * gamma
	Data2 t = (Data2)A * (Data2)gamma;
	
	// qhat = floor((A * gmma) / 2^K)
	Data qhat = (Data)(t >> K);
	
	// p = qhat * q
	Dataplus p = (Dataplus)((Data2)qhat * (Data2)q);
	
	// r = X - p, keep K+1 bits
	Dataplus r = (Dataplus)X - p;
	
	// c = r >> K
	ap_uint<1> c = (ap_uint<1>)(r >> K);
	
	// Z = (r_low + c*delta) mod 2^K
	Z = (Data)(r + (c ? (Dataplus)DELTA : Dataplus(0)));
}

#if TFG_II == 3
// General case: current_stage > 2
void tw_gen_L_s_ge3(const int stage, tapa::ostreams<Data, 3>&  tw_fifo) {
#pragma HLS INLINE off
	
	// const Data L_BASE_s0 = tw_l_base[stage+3];
		
	// R_s0 only valid when stage > 0
	// const Data R_s0 = tw_l_base[stage+2];
	
	// Data L_BASE_s1; reduce(L_BASE_s0, R_s0, L_BASE_s1);
	// Data L_BASE_s2; reduce(L_BASE_s1, R_s0, L_BASE_s2);
	// Data L_BASE_s3; reduce(L_BASE_s2, R_s0, L_BASE_s3);
	const Data L_BASE[3] = {tw_l_base_lane0[stage+3], tw_l_base_lane1[stage+2], tw_l_base_lane2[stage+1]};
#pragma HLS ARRAY_PARTITION variable=L_BASE complete
		
	// Twiddle ratio for this L-stage
	// Data R_s; reduce(tw_l_base[stage+2], tw_l_base[stage+1], R_s);
	// Data R_s = tw_l_Rs[stage];
	Data R_s = tw_l_base_lane1[stage+1];
		
	// Stride for this L-stage
	// const ap_uint<logDEPTH-2> shift = ap_uint<logDEPTH-2>(1 << (num_l_stage - 4 - stage));
		
	// Per-stream state
	Data tw_local[3];
#pragma HLS ARRAY_PARTITION variable=tw_local complete
	tw_local[0] = L_BASE[0];
	tw_local[1] = L_BASE[1];
	tw_local[2] = L_BASE[2];
	// tw_local[3] = L_BASE[3];
		
	// ap_uint<logDEPTH-3> read_idx = 0;
		
	// gamma = floor(R_s * 2^K / q), precomputed offline.
	Data gamma = tw_l_gamma[stage];
	
	const ap_uint<logDEPTH-2> shift1 = ap_uint<logDEPTH-2>(1 << (num_l_stage - 4 - stage));
	ap_uint<logDEPTH-1> read_idx0 = 0;
	ap_uint<logDEPTH-1> read_idx1 = (ap_uint<logDEPTH-1>)shift1;
	ap_uint<logDEPTH-1> read_idx2 = (ap_uint<logDEPTH-1>)(shift1 + shift1);
	const ap_uint<logDEPTH-1> shift2 =  (ap_uint<logDEPTH-1>)(shift1 + shift1);
	const ap_uint<logDEPTH-1> shift3 =  shift2 + (ap_uint<logDEPTH-1>)shift1;
	
	ap_uint<logDEPTH-1> read_idx[3];
#pragma HLS ARRAY_PARTITION variable=read_idx complete
	read_idx[0] = read_idx0;
	read_idx[1] = read_idx1;
	read_idx[2] = read_idx2;
	
	for(;;){
#pragma HLS PIPELINE II=3
		Data nxt_val[3];
#pragma HLS ARRAY_PARTITION variable=nxt_val complete
			
		for (int i = 0; i < 3; ++i) {
#pragma HLS UNROLL
			// 1) Write out current twiddle (4 streams)
			tw_fifo[i].write(tw_local[i]);
			
			// 2) Calculate candidate twiddle
			Data tmp_nxt;
			reduce_tw(tw_local[i], R_s, gamma, tmp_nxt);
			nxt_val[i] = tmp_nxt;
			read_idx[i] = read_idx[i] + shift3;
		}
			
		// 3) Update the index, judge what to do next, preceed or reset to base
		// read_idx += shift;			
		// bool do_step_next = read_idx != 0;
			
		for (int i = 0; i < 3; ++i) {
#pragma HLS UNROLL		
		
			// bool do_step_next = read_idx[i] >= shift3;
			bool L_BASE0 = read_idx[i] == 0;
			bool L_BASE1 = read_idx[i] == shift1;
			bool L_BASE2 = read_idx[i] == shift2;
			// tw_local[i] = do_step_next ? nxt_val[i] : ((L_BASE0)?L_BASE[0]: ((L_BASE1)?L_BASE[1]: ((L_BASE2)?L_BASE[2]: (Data)0)));
			tw_local[i] = L_BASE0 ? L_BASE[0]
            				: L_BASE1 ? L_BASE[1]
            				: L_BASE2 ? L_BASE[2]
            				: nxt_val[i];
		}
			
	}

}

// General case: stage >= 3
void tw_merge_ge3(const int stage, tapa::istreams<Data, 3>& tw_fifo, tapa::ostreams<Data, BU>& tw_L) {
#pragma HLS INLINE off

	Data buf0, buf1, buf2;
#pragma HLS RESET variable=buf0
#pragma HLS RESET variable=buf1
#pragma HLS RESET variable=buf2

	ap_uint<2> phase = 0; // 0,1,2

	for (;;) {
#pragma HLS PIPELINE II=1

		// Only read FIFOs once every 3 cycles (phase==0)
		if (phase == 0) {
			buf0 = tw_fifo[0].read();
			buf1 = tw_fifo[1].read();
			buf2 = tw_fifo[2].read();
		}

		Data tw_local = (phase == 0) ? buf0
				: (phase == 1) ? buf1
						: buf2;

		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_local);
		}

		phase = (phase == 2) ? (ap_uint<2>)0 : (ap_uint<2>)(phase + 1);
	}
}

void bf_unit_ge3(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  int global_stage = stage_local + 3;  // local 0,1,2... => global 3,4,...
  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);
}

// Wrapper for num_l_stage > 3
void l_stage_s_ge3(int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
#pragma HLS INLINE off
	// L-stage index = 3, 4, ...
	
	tapa::streams<Data, 3, 3> tw_fifo("tw_fifo");
	
	tapa::streams<Data, BU, 2> tw_L("twL");

	// generators (II=3 each)
	tapa::task()
		.invoke<tapa::detach>(tw_gen_L_s_ge3, stage, tw_fifo) // A 4-lane twiddle generator (inner II=3) 
		.invoke<tapa::detach>(tw_merge_ge3, stage, tw_fifo, tw_L) // Merger: tw_fifo[0]-[3] -> BU tw_L[j] (overall II=1)
		.invoke<tapa::detach, BU>(bf_unit_ge3, stage, tapa::seq(), input_stream, output_stream, tw_L); // NBU bf_units, keeping II=1
}

// Special case: stage == 2, where four streams involves constant tw_base streams
void tw_gen_L_s2(const int stage, tapa::ostreams<Data, 4>&  tw_fifo) {
#pragma HLS INLINE off
	
	// const Data L_BASE_s0 = tw_l_base[stage];
	
	// R_s0 only valid when stage > 0
	// const Data R_s0 = tw_l_base[stage-1];
	
	// Data L_BASE_s1; reduce(L_BASE_s0, R_s0, L_BASE_s1);
	// Data L_BASE_s2; reduce(L_BASE_s1, R_s0, L_BASE_s2);
	// Data L_BASE_s3; reduce(L_BASE_s2, R_s0, L_BASE_s3);
	// const Data L_BASE[4] = {L_BASE_s0, L_BASE_s1, L_BASE_s2, L_BASE_s3};
	const Data L_BASE[4] = {tw_l_base_lane0[2], tw_l_base_lane1[1], tw_l_base_lane2[0], tw_l_base_lane3[0]};
#pragma HLS ARRAY_PARTITION variable=L_BASE complete

	for(;;){
#pragma HLS PIPELINE II=3
		for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
			tw_fifo[i].write(L_BASE[i]);
		}
	}
		
}

void tw_merge_s2(const int stage, tapa::istreams<Data, 4>& tw_fifo, tapa::ostreams<Data, BU>&  tw_L) {
	ap_uint<2> idx = 0;
	Data tw_local;
	for(;;){
#pragma HLS PIPELINE II=1
		// Alternately read from tw_fifo[0] - tw_fifo[3]
		tw_local = tw_fifo[idx].read();
		
		// Broadcast NBU twiddles to NBU butterfly unit
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
				tw_L[j].write(tw_local);
			}
		idx++;
	}
}

void bf_unit_s2(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}

void l_stage_s2(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
	// 4-lane twiddle stream:
	tapa::streams<Data, 4, 2> tw_fifo("tw_fifo");
	
	tapa::streams<Data, BU, 2> tw_L("twL");

	// generators (II=4 each)
	tapa::task()
		.invoke<tapa::detach>(tw_gen_L_s2, stage, tw_fifo) // A 4-lane twiddle generator (inner II=4) 
		.invoke<tapa::detach>(tw_merge_s2, stage, tw_fifo, tw_L) // Merger: tw_fifo[0]-[3] -> BU tw_L[j] (overall II=1)
		.invoke<tapa::detach, BU>(bf_unit_s2, stage, tapa::seq(), input_stream, output_stream, tw_L); // NBU bf_units, keeping II=1
}
#else
#if TFG_II == 1
// General case: current_stage > 1
void tw_gen_L_s_ge2(const int stage, tapa::ostream<Data>&  tw_fifo) {
#pragma HLS INLINE off
	
	// const Data L_BASE_s0 = tw_l_base[stage+2];
	const Data L_BASE_s0 = tw_l_base_lane0[stage+2];	
	// Twiddle ratio for this L-stage
	// const Data R_s = tw_l_base[stage+1];
	const Data R_s = tw_l_base_lane0[stage+1];	
	// Stride for this L-stage
	const ap_uint<logDEPTH-1> shift = ap_uint<logDEPTH-2>(1 << (num_l_stage - 3 - stage));		
	// Per-stream state
	Data tw_local = L_BASE_s0;		
	ap_uint<logDEPTH-1> read_idx = 0;
	
	// gamma = floor(R_s * 2^K / q), precomputed offline.
	Data gamma = tw_l_gamma[stage];
	
	for(;;){
#pragma HLS PIPELINE II=1
		Data nxt_val;		
		// 1) Write out current twiddle (2 streams)
		tw_fifo.write(tw_local);		
		// 2) Calculate candidate twiddle
		Data tmp_nxt;
		reduce_tw(tw_local, R_s, gamma, tmp_nxt);
		nxt_val = tmp_nxt;							
		// 3) Update the index, judge what to do next, preceed or reset to base
		read_idx += shift;			
		bool do_step_next = read_idx != 0;				
		tw_local = do_step_next ? nxt_val : L_BASE_s0;			
	}
}

// General case: stage >= 2
void tw_merge_ge2(const int stage, tapa::istream<Data>& tw_fifo, tapa::ostreams<Data, BU>&  tw_L) {
	Data tw_local;
	for(;;){
#pragma HLS PIPELINE II=1
		// Alternately read from tw_fifo[0] / tw_fifo[1]
		tw_local = tw_fifo.read();
		
		// Broadcast NBU twiddles to NBU butterfly unit
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(tw_local);
		}
	}
}

#elif TFG_II == 2
// General case: current_stage > 1
void tw_gen_L_s_ge2(const int stage, tapa::ostreams<Data, 2>&  tw_fifo) {
#pragma HLS INLINE off
	
	//const Data L_BASE_s0 = tw_l_base[stage+2];
		
	// R_s0 only valid when stage > 0
	// const Data R_s0 = tw_l_base[stage+1];
	// Compute second base for odd stream: L_BASE_s1 = L_BASE_s0 * R_s0 (mod q)
	// Data nxt_base; reduce(L_BASE_s0, R_s0, nxt_base);
	
	//const Data L_BASE[2] = {L_BASE_s0, nxt_base};
	const Data L_BASE[2] = {tw_l_base_lane0[stage+2], tw_l_base_lane1[stage+1]};
#pragma HLS ARRAY_PARTITION variable=L_BASE complete
		
	// Twiddle ratio for this L-stage
	const Data R_s = tw_l_base_lane0[stage];
		
	// Stride for this L-stage
	const ap_uint<logDEPTH-2> shift = ap_uint<logDEPTH-2>(1 << (num_l_stage - 3 - stage));
		
	// Per-stream state
	Data tw_local[2];
#pragma HLS ARRAY_PARTITION variable=tw_local complete
	tw_local[0] = L_BASE[0]; // even stream
	tw_local[1] = L_BASE[1]; // odd stream
		
	ap_uint<logDEPTH-2> read_idx = 0;
		
	Data gamma = tw_l_gamma[stage];
	
	for(;;){
#pragma HLS PIPELINE II=2
		Data nxt_val[2];
#pragma HLS ARRAY_PARTITION variable=nxt_val complete
			
		for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
			// 1) Write out current twiddle (2 streams)
			tw_fifo[i].write(tw_local[i]);
			
			// 2) Calculate candidate twiddle
			Data tmp_nxt;
			// reduce(tw_local[i], R_s, tmp_nxt);
			reduce_tw(tw_local[i], R_s, gamma, tmp_nxt);
			nxt_val[i] = tmp_nxt;	
		}			
			
		// 3) Update the index, judge what to do next, preceed or reset to base
		read_idx += shift;			
		bool do_step_next = read_idx != 0;			
		for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL		
			tw_local[i] = do_step_next ? nxt_val[i] : L_BASE[i];
		}
			
	}

}

// General case: stage >= 2
void tw_merge_ge2(const int stage, tapa::istreams<Data, 2>& tw_fifo, tapa::ostreams<Data, BU>&  tw_L) {
	ap_uint<1> idx = 0;
	Data tw_local;
	for(;;){
#pragma HLS PIPELINE II=1
		// Alternately read from tw_fifo[0] / tw_fifo[1]
		tw_local = tw_fifo[idx].read();
		
		// Broadcast NBU twiddles to NBU butterfly unit
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
				tw_L[j].write(tw_local);
			}
		idx++;
	}
}
#endif

void bf_unit_ge2(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  int global_stage = stage_local + 2;  // local 0,1,... => global 2,3,...
  bf_unit(global_stage, bf_id, input_stream, output_stream, tw_i);
}

// Wrapper for num_l_stage > 2
void l_stage_s_ge2(int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
#pragma HLS INLINE off
	// L-stage index = 2, 3, 4, ...
	// 2-lane twiddle stream: [0] = even, [1] = odd
	tapa::streams<Data, 2, 2> tw_fifo("tw_fifo");
	
	tapa::streams<Data, BU, 2> tw_L("twL");

	// generators (II=2 each)
	tapa::task()
		.invoke<tapa::detach>(tw_gen_L_s_ge2, stage, tw_fifo) // A 2-lane twiddle generator (inner II=2) 
		.invoke<tapa::detach>(tw_merge_ge2, stage, tw_fifo, tw_L) // Merger: tw_fifo[0]/[1] -> BU tw_L[j] (overall II=1)
		.invoke<tapa::detach, BU>(bf_unit_ge2, stage, tapa::seq(), input_stream, output_stream, tw_L); // NBU bf_units, keeping II=1
}
#endif

// Special case: stage == 1, where two streams involves constant tw_base streams
void tw_gen_L_s1(const int stage, tapa::ostreams<Data, 2>&  tw_fifo) {
#pragma HLS INLINE off
	
	// const Data L_BASE_s0 = tw_l_base[stage];
	
	// R_s0 only valid when stage > 0
	// const Data R_s0 = tw_l_base[stage-1];
	// Compute second base for odd stream: L_BASE_s1 = L_BASE_s0 * R_s0 (mod q)
	// Data nxt_base; reduce(L_BASE_s0, R_s0, nxt_base);
	// const Data L_BASE[2] = {L_BASE_s0, nxt_base};
	const Data L_BASE[2] = {tw_l_base_lane0[1], tw_l_base_lane1[0]};
#pragma HLS ARRAY_PARTITION variable=L_BASE complete

	for(;;){
#pragma HLS PIPELINE II=2
		for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
			tw_fifo[i].write(L_BASE[i]);
		}
	}
		
}

void tw_merge_s1(const int stage, tapa::istreams<Data, 2>& tw_fifo, tapa::ostreams<Data, BU>&  tw_L) {
	ap_uint<1> idx = 0;
	Data tw_local;
	for(;;){
#pragma HLS PIPELINE II=1
		// Alternately read from tw_fifo[0] / tw_fifo[1]
		tw_local = tw_fifo[idx].read();
		
		// Broadcast NBU twiddles to NBU butterfly unit
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
				tw_L[j].write(tw_local);
			}
		idx++;
	}
}

void bf_unit_s1(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}

void l_stage_s1(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
	// 2-lane twiddle stream: [0] = even, [1] = odd
	tapa::streams<Data, 2, 2> tw_fifo("tw_fifo");
	
	tapa::streams<Data, BU, 2> tw_L("twL");

	// generators (II=2 each)
	tapa::task()
		.invoke<tapa::detach>(tw_gen_L_s1, stage, tw_fifo) // A 2-lane twiddle generator (inner II=2) 
		.invoke<tapa::detach>(tw_merge_s1, stage, tw_fifo, tw_L) // Merger: tw_fifo[0]/[1] -> BU tw_L[j] (overall II=1)
		.invoke<tapa::detach, BU>(bf_unit_s1, stage, tapa::seq(), input_stream, output_stream, tw_L); // NBU bf_units, keeping II=1
}

// Special case: stage == 0, where two streams always output tw_l_base[0]
void tw_gen_L_s0(const int stage, tapa::ostreams<Data, BU>&  tw_L) {
#pragma HLS INLINE off
	
	const Data L_BASE_s0 = tw_l_base_lane0[0];
	
	for(;;){ 
#pragma HLS PIPELINE II=1
		for (int j = 0; j < BU; ++j) {
#pragma HLS UNROLL
			tw_L[j].write(L_BASE_s0);
		}
	}
	
}

void bf_unit_s0(int stage_local, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream, tapa::istream<Data>& tw_i) {
#pragma HLS INLINE off
  bf_unit(stage_local, bf_id, input_stream, output_stream, tw_i);
}

void l_stage_s0(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){
	
	tapa::streams<Data, BU, 2> tw_L("twL");

	tapa::task()
		.invoke<tapa::detach>(tw_gen_L_s0, stage, tw_L)
		.invoke<tapa::detach, BU>(bf_unit_s0, stage, tapa::seq(), input_stream, output_stream, tw_L);
}

void tw_gen_X(tapa::ostreams<Wide, num_x_stage>& tw_X_W) {
		
	Data tw_local[num_x_stage][BU];
#pragma HLS ARRAY_PARTITION variable=tw_local dim=1 complete
#pragma HLS ARRAY_PARTITION variable=tw_local dim=2 complete
	Data R_s[num_x_stage];
#pragma HLS ARRAY_PARTITION variable=R_s complete

	ap_uint<logDEPTH> i = 1;

	tw_local[0][0] = tw_x_base[0];
	R_s[0] = tw_l_base_lane0[num_l_stage-1];
	
	for(int s = 1; s < num_x_stage; s++){
#pragma HLS UNROLL
		const int num_tw_base = 1 << s;
		const int base_offset = (1 << s) - 1;
		for(int g = 0; g < num_tw_base; ++g) {
#pragma HLS UNROLL
			tw_local[s][g] = tw_x_base[base_offset+g];
		}
		R_s[s] = tw_x_base[(1<<(s-1))-1];
	}

	for(;;){
#pragma HLS PIPELINE II = 1

STAGE_LOOP:
		for(int s = 0; s < num_x_stage; s++){
#pragma HLS UNROLL
			const int num_tw_base = 1 << s;
			const int shift = logBU - s;
			const int base_offset = (1 << s) - 1;
			
			Wide w = 0;
DISTRIBUTION_LOOP:
			for(int idx = 0; idx < BU; idx++){
#pragma HLS UNROLL
				const int group_idx = idx >> shift;
				w.range((idx+1)*K-1, idx*K) = tw_local[s][group_idx];       
			}
			tw_X_W[s].write(w);
						
TW_MUL_MOD_LOOP:			
			for(int g = 0; g < num_tw_base; ++g) {
#pragma HLS UNROLL
				Data nxt; reduce(tw_local[s][g], R_s[s], nxt);
				tw_local[s][g] = (i)?nxt:tw_x_base[base_offset+g];
			}
				

		}
		i++;
	}

}

void x_stages(tapa::istreams<Data2, BU>& input_streams, tapa::ostreams<Data2, BU>& output_streams, tapa::istreams<Wide, num_x_stage>& tw_X_W){

	const int num_of_stages = logBU + 1;
	Data mem[num_of_stages+1][WIDTH];
	ap_uint<logDEPTH> i = 0;
	
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

					butterfly(mem[s][2*idx], mem[s][2*idx+1], twf, &out_even, &out_odd);
					mem[s+1][ind_even] = out_even;
					mem[s+1][ind_odd] = out_odd;
						
					            
				}
				
			}

OUTPUT_LOOP:
			for(int j = 0; j < BU; j++){
#pragma HLS UNROLL
				Data2 out_data = (mem[num_of_stages][2*j+1], mem[num_of_stages][2*j]);
				output_streams[j].write(out_data);
			}

			i++;
		}
	}
}

void x_stages_top(tapa::istreams<Data2, BU>& input_streams, tapa::ostreams<Data2, BU>& output_streams){
	tapa::streams<Wide, num_x_stage, 2> tw_X_W("twX");

	tapa::task()
		.invoke<tapa::detach>(tw_gen_X, tw_X_W)
		.invoke<tapa::detach>(x_stages, input_streams, output_streams, tw_X_W);
}

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

#ifdef MCH

void read_dram_m(tapa::mmap<bits<DataVec>> x, tapa::ostreams<Data, 2*BU> & dramrd_streams, int poly_num){

	for(int poly = 0; poly < poly_num/CH; poly++){
        	for(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
			DataVec data = tapa::bit_cast<DataVec>( x[poly * (n/DataCHLen) + i] ); 
			for(int d = 0; d < DataCHLen; d++){
#pragma HLS UNROLL
				dramrd_streams[(i % CORE_DRAM_WIDTH_RATIO)*DataCHLen+d].write( data[d] );			
			}
		}	
	}
}

void read_collect_2_core_m(tapa::istream<Data> & dramrd_streams0_e, tapa::istream<Data> & dramrd_streams1_e, tapa::istream<Data> & dramrd_streams0_o, tapa::istream<Data> & dramrd_streams1_o, tapa::ostream<Data2> & core_istream){

READ_COLLECT_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data data_e;
				Data data_o;
				if( ch == 0 ){
					data_e = dramrd_streams0_e.read();
					data_o = dramrd_streams0_o.read();
				}
				else{
					data_e = dramrd_streams1_e.read();
					data_o = dramrd_streams1_o.read();
				}

				Data2 data = (data_o, data_e);
				core_istream.write( data );
			}
		}
	}
}

void read_collect_4_core_m(tapa::istream<Data> & dramrd_streams0_e, tapa::istream<Data> & dramrd_streams1_e, tapa::istream<Data> & dramrd_streams2_e, tapa::istream<Data> & dramrd_streams3_e, tapa::istream<Data> & dramrd_streams0_o, tapa::istream<Data> & dramrd_streams1_o, tapa::istream<Data> & dramrd_streams2_o, tapa::istream<Data> & dramrd_streams3_o, tapa::ostream<Data2> & core_istream){

READ_COLLECT_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data data_e;
				Data data_o;
				if( ch == 0 ){
					data_e = dramrd_streams0_e.read();
					data_o = dramrd_streams0_o.read();
				}
				else if( ch == 1 ){
					data_e = dramrd_streams1_e.read();
					data_o = dramrd_streams1_o.read();
				}
				else if( ch == 2 ){
					data_e = dramrd_streams2_e.read();
					data_o = dramrd_streams2_o.read();
				}
				else{
					data_e = dramrd_streams3_e.read();
					data_o = dramrd_streams3_o.read();
				}

				Data2 data = (data_o, data_e);
				core_istream.write( data );
			}
		}
	}
}

void read_collect_8_core_m(tapa::istream<Data> & dramrd_streams0_e, tapa::istream<Data> & dramrd_streams1_e, tapa::istream<Data> & dramrd_streams2_e, tapa::istream<Data> & dramrd_streams3_e, tapa::istream<Data> & dramrd_streams4_e, tapa::istream<Data> & dramrd_streams5_e, tapa::istream<Data> & dramrd_streams6_e, tapa::istream<Data> & dramrd_streams7_e, tapa::istream<Data> & dramrd_streams0_o, tapa::istream<Data> & dramrd_streams1_o, tapa::istream<Data> & dramrd_streams2_o, tapa::istream<Data> & dramrd_streams3_o, tapa::istream<Data> & dramrd_streams4_o, tapa::istream<Data> & dramrd_streams5_o, tapa::istream<Data> & dramrd_streams6_o, tapa::istream<Data> & dramrd_streams7_o, tapa::ostream<Data2> & core_istream){

READ_COLLECT_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data data_e;
				Data data_o;
				if( ch == 0 ){
					data_e = dramrd_streams0_e.read();
					data_o = dramrd_streams0_o.read();
				}
				else if( ch == 1 ){
					data_e = dramrd_streams1_e.read();
					data_o = dramrd_streams1_o.read();
				}
				else if( ch == 2 ){
					data_e = dramrd_streams2_e.read();
					data_o = dramrd_streams2_o.read();
				}
				else if( ch == 3 ){
					data_e = dramrd_streams3_e.read();
					data_o = dramrd_streams3_o.read();
				}
				else if( ch == 4 ){
					data_e = dramrd_streams4_e.read();
					data_o = dramrd_streams4_o.read();
				}
				else if( ch == 5 ){
					data_e = dramrd_streams5_e.read();
					data_o = dramrd_streams5_o.read();
				}
				else if( ch == 6 ){
					data_e = dramrd_streams6_e.read();
					data_o = dramrd_streams6_o.read();
				}
				else{
					data_e = dramrd_streams7_e.read();
					data_o = dramrd_streams7_o.read();
				}

				Data2 data = (data_o, data_e);
				core_istream.write( data );
			}
		}
	}
}

void read_collect_2_m(tapa::istreams<Data, BU> & dramrd_streams0_e, tapa::istreams<Data, BU> & dramrd_streams0_o, tapa::istreams<Data, BU> & dramrd_streams1_e, tapa::istreams<Data, BU> & dramrd_streams1_o, tapa::ostreams<Data2, BU> & core_istreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(read_collect_2_core_m, dramrd_streams0_e, dramrd_streams1_e, dramrd_streams0_o, dramrd_streams1_o, core_istreams)
	;
}

void read_collect_4_m(tapa::istreams<Data, BU> & dramrd_streams0_e, tapa::istreams<Data, BU> & dramrd_streams0_o, tapa::istreams<Data, BU> & dramrd_streams1_e, tapa::istreams<Data, BU> & dramrd_streams1_o, tapa::istreams<Data, BU> & dramrd_streams2_e, tapa::istreams<Data, BU> & dramrd_streams2_o, tapa::istreams<Data, BU> & dramrd_streams3_e, tapa::istreams<Data, BU> & dramrd_streams3_o, tapa::ostreams<Data2, BU> & core_istreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(read_collect_4_core_m, dramrd_streams0_e, dramrd_streams1_e, dramrd_streams2_e, dramrd_streams3_e, dramrd_streams0_o, dramrd_streams1_o, dramrd_streams2_o, dramrd_streams3_o, core_istreams)
	;
}

void read_collect_8_m(tapa::istreams<Data, BU> & dramrd_streams0_e, tapa::istreams<Data, BU> & dramrd_streams0_o, tapa::istreams<Data, BU> & dramrd_streams1_e, tapa::istreams<Data, BU> & dramrd_streams1_o, tapa::istreams<Data, BU> & dramrd_streams2_e, tapa::istreams<Data, BU> & dramrd_streams2_o, tapa::istreams<Data, BU> & dramrd_streams3_e, tapa::istreams<Data, BU> & dramrd_streams3_o, tapa::istreams<Data, BU> & dramrd_streams4_e, tapa::istreams<Data, BU> & dramrd_streams4_o, tapa::istreams<Data, BU> & dramrd_streams5_e, tapa::istreams<Data, BU> & dramrd_streams5_o, tapa::istreams<Data, BU> & dramrd_streams6_e, tapa::istreams<Data, BU> & dramrd_streams6_o, tapa::istreams<Data, BU> & dramrd_streams7_e, tapa::istreams<Data, BU> & dramrd_streams7_o, tapa::ostreams<Data2, BU> & core_istreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(read_collect_8_core_m, dramrd_streams0_e, dramrd_streams1_e, dramrd_streams2_e, dramrd_streams3_e, dramrd_streams4_e, dramrd_streams5_e, dramrd_streams6_e, dramrd_streams7_e, dramrd_streams0_o, dramrd_streams1_o, dramrd_streams2_o, dramrd_streams3_o, dramrd_streams4_o, dramrd_streams5_o, dramrd_streams6_o, dramrd_streams7_o, core_istreams)
	;
}

void read_collect_m(tapa::istreams<Data, 2*BU*GROUP_CH_NUM> & dramrd_streams, tapa::ostreams<Data2, BU> & core_istreams ){

	tapa::task()
#if GROUP_CH_NUM == 2
		.invoke<tapa::detach, 1>(read_collect_2_m, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, core_istreams)
#elif GROUP_CH_NUM == 4
		.invoke<tapa::detach, 1>(read_collect_4_m, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, core_istreams)
#elif GROUP_CH_NUM == 8
		.invoke<tapa::detach, 1>(read_collect_8_m, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, dramrd_streams, core_istreams)
#endif
	;
}

void write_dist_2_core_m(tapa::ostreams<Data,2> & dramwr_streams0, tapa::ostreams<Data,2> & dramwr_streams1, tapa::istream<Data2> & core_ostream){

WRITE_DIST_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data2 data = core_ostream.read();
				Data data_e = data(K-1,0);
				Data data_o = data(2*K-1,K);

				if( ch == 0 ){ 
					dramwr_streams0[0].write( data_e );
					dramwr_streams0[1].write( data_o );
				}
				else{ 
					dramwr_streams1[0].write( data_e );
					dramwr_streams1[1].write( data_o );
				}
			}
		}
	}
}

void write_dist_4_core_m(tapa::ostreams<Data,2> & dramwr_streams0, tapa::ostreams<Data,2> & dramwr_streams1, tapa::ostreams<Data,2> & dramwr_streams2, tapa::ostreams<Data,2> & dramwr_streams3, tapa::istream<Data2> & core_ostream){

WRITE_DIST_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data2 data = core_ostream.read();
				Data data_e = data(K-1,0);
				Data data_o = data(2*K-1,K);

				if( ch == 0 ){ 
					dramwr_streams0[0].write( data_e );
					dramwr_streams0[1].write( data_o );
				}
				else if( ch == 1 ){ 
					dramwr_streams1[0].write( data_e );
					dramwr_streams1[1].write( data_o );
				}
				else if( ch == 2 ){ 
					dramwr_streams2[0].write( data_e );
					dramwr_streams2[1].write( data_o );
				}
				else{ 
					dramwr_streams3[0].write( data_e );
					dramwr_streams3[1].write( data_o );
				}
			}
		}
	}
}

void write_dist_8_core_m(tapa::ostreams<Data,2> & dramwr_streams0, tapa::ostreams<Data,2> & dramwr_streams1, tapa::ostreams<Data,2> & dramwr_streams2, tapa::ostreams<Data,2> & dramwr_streams3, tapa::ostreams<Data,2> & dramwr_streams4, tapa::ostreams<Data,2> & dramwr_streams5, tapa::ostreams<Data,2> & dramwr_streams6, tapa::ostreams<Data,2> & dramwr_streams7, tapa::istream<Data2> & core_ostream){

WRITE_DIST_LOOP:
	for(;;){		
		for(int ch = 0; ch < GROUP_CH_NUM; ch++){
			for(int i = 0; i < n/(2*BU); i++){
#pragma HLS PIPELINE II = 1
				Data2 data = core_ostream.read();
				Data data_e = data(K-1,0);
				Data data_o = data(2*K-1,K);

				if( ch == 0 ){ 
					dramwr_streams0[0].write( data_e );
					dramwr_streams0[1].write( data_o );
				}
				else if( ch == 1 ){ 
					dramwr_streams1[0].write( data_e );
					dramwr_streams1[1].write( data_o );
				}
				else if( ch == 2 ){ 
					dramwr_streams2[0].write( data_e );
					dramwr_streams2[1].write( data_o );
				}
				else if( ch == 3 ){ 
					dramwr_streams3[0].write( data_e );
					dramwr_streams3[1].write( data_o );
				}
				else if( ch == 4 ){ 
					dramwr_streams4[0].write( data_e );
					dramwr_streams4[1].write( data_o );
				}
				else if( ch == 5 ){ 
					dramwr_streams5[0].write( data_e );
					dramwr_streams5[1].write( data_o );
				}
				else if( ch == 6 ){ 
					dramwr_streams6[0].write( data_e );
					dramwr_streams6[1].write( data_o );
				}
				else{ 
					dramwr_streams7[0].write( data_e );
					dramwr_streams7[1].write( data_o );
				}
			}
		}
	}
}

void write_dist_2_m(tapa::ostreams<Data, 2*BU> & dramwr_streams0, tapa::ostreams<Data, 2*BU> & dramwr_streams1, tapa::istreams<Data2, BU> & core_ostreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(write_dist_2_core_m, dramwr_streams0, dramwr_streams1, core_ostreams)
	;
}

void write_dist_4_m(tapa::ostreams<Data, 2*BU> & dramwr_streams0, tapa::ostreams<Data, 2*BU> & dramwr_streams1, tapa::ostreams<Data, 2*BU> & dramwr_streams2, tapa::ostreams<Data, 2*BU> & dramwr_streams3, tapa::istreams<Data2, BU> & core_ostreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(write_dist_4_core_m, dramwr_streams0, dramwr_streams1, dramwr_streams2, dramwr_streams3, core_ostreams)
	;
}

void write_dist_8_m(tapa::ostreams<Data, 2*BU> & dramwr_streams0, tapa::ostreams<Data, 2*BU> & dramwr_streams1, tapa::ostreams<Data, 2*BU> & dramwr_streams2, tapa::ostreams<Data, 2*BU> & dramwr_streams3, tapa::ostreams<Data, 2*BU> & dramwr_streams4, tapa::ostreams<Data, 2*BU> & dramwr_streams5, tapa::ostreams<Data, 2*BU> & dramwr_streams6, tapa::ostreams<Data, 2*BU> & dramwr_streams7, tapa::istreams<Data2, BU> & core_ostreams ){

	tapa::task()
		.invoke<tapa::detach, BU>(write_dist_8_core_m, dramwr_streams0, dramwr_streams1, dramwr_streams2, dramwr_streams3, dramwr_streams4, dramwr_streams5, dramwr_streams6, dramwr_streams7, core_ostreams)
	;
}

void write_dist_m(tapa::ostreams<Data, 2*BU*GROUP_CH_NUM> & dramwr_streams, tapa::istreams<Data2, BU> & core_ostreams ){

	tapa::task()
#if GROUP_CH_NUM == 2
		.invoke<tapa::detach, 1>(write_dist_2_m, dramwr_streams, dramwr_streams, core_ostreams)
#elif GROUP_CH_NUM == 4
		.invoke<tapa::detach, 1>(write_dist_4_m, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, core_ostreams)
#elif GROUP_CH_NUM == 8
		.invoke<tapa::detach, 1>(write_dist_8_m, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, dramwr_streams, core_ostreams)
#endif
	;
}

void write_dram_m(tapa::mmap<bits<DataVec>> y, tapa::istreams<Data, 2*BU> & dramwr_streams, int poly_num){

	for(int poly = 0; poly < poly_num/CH; poly++){
        	for(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
			DataVec data;
			for(int d = 0; d < DataCHLen; d++){
#pragma HLS UNROLL
				data[d] = dramwr_streams[(i % CORE_DRAM_WIDTH_RATIO)*DataCHLen+d].read();			
			}
            		y[poly * (n/DataCHLen) + i] =  tapa::bit_cast<bits<DataVec>>(data);
		}	
	}
}

#else //MCH

void read_dram_s(tapa::mmap<bits<DataVec>> x, tapa::ostreams<DataVec, GROUP_CORE_NUM> & dramrd_streams, int poly_num){

	for(int poly = 0; poly < poly_num/CH; poly++){
	       	for(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
			DataVec data = tapa::bit_cast<DataVec>( x[poly * (n/DataCHLen) + i] );
			dramrd_streams[poly%GROUP_CORE_NUM].write(data);
		}
	}
}

void read_dist_s(tapa::istream<DataVec> & dramrd_stream, tapa::ostreams<Data2, BU> & core_istreams){

READ_DIST_S_LOOP:
	for(;;){
#pragma HLS PIPELINE II = DRAM_CORE_WIDTH_RATIO
		if( !dramrd_stream.empty() ){
			DataVec datavec = dramrd_stream.read();
			for(int core = 0; core < DRAM_CORE_WIDTH_RATIO; core++){
				for(int d = 0; d < BU; d++){
#pragma HLS UNROLL	
					Data2 data = ( static_cast<Data>(datavec[core*2*BU+BU+d]), static_cast<Data>(datavec[core*2*BU+d]) );
					core_istreams[d].write( data );
				}
			}
		}
	}
}

void write_reshape_s(tapa::ostream<DataVec> & dramwr_stream, tapa::istreams<Data2, BU> & core_ostreams ){

WRITE_RESHAPE_S_LOOP:
	for(;;){
#pragma HLS PIPELINE II = DRAM_CORE_WIDTH_RATIO
		DataVec datavec;
		for(int c = 0; c < DRAM_CORE_WIDTH_RATIO; c++){
			for(int d = 0; d < BU; d++){
				Data2 data = core_ostreams[d].read();
				Data data_e = data(K-1,0);
				Data data_o = data(2*K-1,K);
				datavec[c*2*BU+2*d] = static_cast<HostData>(data_e);
				datavec[c*2*BU+2*d+1] = static_cast<HostData>(data_o);
			}
		}
		dramwr_stream.write(datavec);
	}
}

void write_dram_s(tapa::mmap<bits<DataVec>> y, tapa::istreams<DataVec, GROUP_CORE_NUM> & dramwr_streams, int poly_num){

	for(int poly = 0; poly < poly_num/CH; poly++){
	       	for(int i = 0; i < n/DataCHLen; i++){
#pragma HLS PIPELINE II = 1
			DataVec data = dramwr_streams[poly%GROUP_CORE_NUM].read();
            		y[poly * (n/DataCHLen) + i] =  tapa::bit_cast<bits<DataVec>>(data);
		}
	}
}

#endif //MCH


void ntt_core(tapa::istreams<Data2, BU> core_istreams, tapa::ostreams<Data2, BU> core_ostreams){

	tapa::streams<Data2, BU*(num_l_stage+1), 2> core_streams("core_streams");

	tapa::task()
		.invoke<tapa::detach, BU>(input_mem_stage, core_istreams, core_streams)
		.invoke<tapa::detach>(l_stage_s0, 0, core_streams, core_streams)
		.invoke<tapa::detach>(l_stage_s1, 1, core_streams, core_streams)
#if NUM_L_stage > 2 // num_l_stage > 2
#if TFG_II == 3		
		.invoke<tapa::detach>(l_stage_s2, 2, core_streams, core_streams)
		.invoke<tapa::detach, num_l_stage_ge3>(l_stage_s_ge3, tapa::seq(), core_streams, core_streams) // After splitting, the number of batches should minus 3
#else
		.invoke<tapa::detach, num_l_stage_ge2>(l_stage_s_ge2, tapa::seq(), core_streams, core_streams) // After splitting, the number of batches should minus 2
#endif
#endif
		.invoke<tapa::detach>(x_stages_top, core_streams, core_ostreams); 
}

#ifndef MCH

void ntt_group_dram1(tapa::mmap<bits<DataVec>> x, tapa::mmap<bits<DataVec>> y, int poly_num){

	tapa::streams<DataVec, GROUP_CORE_NUM, POLY_FIFO_DEPTH_S> dramrd_streams("dramrd_streams");
	tapa::streams<DataVec, GROUP_CORE_NUM, POLY_FIFO_DEPTH_S> dramwr_streams("dramwr_streams");
	tapa::streams<Data2, BU*GROUP_CORE_NUM, 2> core_istreams("core_istreams");
	tapa::streams<Data2, BU*GROUP_CORE_NUM, 2> core_ostreams("core_ostreams");

	tapa::task()
		.invoke<tapa::join>(read_dram_s, x, dramrd_streams, poly_num)
	 	.invoke<tapa::detach, GROUP_CORE_NUM>(read_dist_s, dramrd_streams, core_istreams)

		.invoke<tapa::detach, GROUP_CORE_NUM>(ntt_core, core_istreams, core_ostreams)

		.invoke<tapa::detach, GROUP_CORE_NUM>(write_reshape_s, dramwr_streams, core_ostreams)
		.invoke<tapa::join>(write_dram_s, y, dramwr_streams, poly_num)
	;
}

#else //MCH

void ntt_group_dram2(
	tapa::mmap<bits<DataVec>> x0, 
	tapa::mmap<bits<DataVec>> x1, 
	tapa::mmap<bits<DataVec>> y0, 
	tapa::mmap<bits<DataVec>> y1, 
	int poly_num){

	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramrd_streams("dramrd_streams");
	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramwr_streams("dramwr_streams");
	tapa::streams<Data2, BU, 2> core_istreams("core_istreams");
	tapa::streams<Data2, BU, 2> core_ostreams("core_ostreams");

	tapa::task()
		.invoke<tapa::join>(read_dram_m, x0, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x1, dramrd_streams, poly_num)
		.invoke<tapa::detach>(read_collect_m, dramrd_streams, core_istreams)

		.invoke<tapa::detach>(ntt_core, core_istreams, core_ostreams)

		.invoke<tapa::detach>(write_dist_m, dramwr_streams, core_ostreams)
		.invoke<tapa::join>(write_dram_m, y0, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y1, dramwr_streams, poly_num)
	;
}

void ntt_group_dram4(
	tapa::mmap<bits<DataVec>> x0, 
	tapa::mmap<bits<DataVec>> x1, 
	tapa::mmap<bits<DataVec>> x2, 
	tapa::mmap<bits<DataVec>> x3, 
	tapa::mmap<bits<DataVec>> y0, 
	tapa::mmap<bits<DataVec>> y1, 
	tapa::mmap<bits<DataVec>> y2, 
	tapa::mmap<bits<DataVec>> y3, 
	int poly_num){

	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramrd_streams("dramrd_streams");
	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramwr_streams("dramwr_streams");
	tapa::streams<Data2, BU, 2> core_istreams("core_istreams");
	tapa::streams<Data2, BU, 2> core_ostreams("core_ostreams");

	tapa::task()
		.invoke<tapa::join>(read_dram_m, x0, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x1, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x2, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x3, dramrd_streams, poly_num)
		.invoke<tapa::detach>(read_collect_m, dramrd_streams, core_istreams)

		.invoke<tapa::detach>(ntt_core, core_istreams, core_ostreams)

		.invoke<tapa::detach>(write_dist_m, dramwr_streams, core_ostreams)
		.invoke<tapa::join>(write_dram_m, y0, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y1, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y2, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y3, dramwr_streams, poly_num)
	;
}

void ntt_group_dram8(
	tapa::mmap<bits<DataVec>> x0, 
	tapa::mmap<bits<DataVec>> x1, 
	tapa::mmap<bits<DataVec>> x2, 
	tapa::mmap<bits<DataVec>> x3, 
	tapa::mmap<bits<DataVec>> x4, 
	tapa::mmap<bits<DataVec>> x5, 
	tapa::mmap<bits<DataVec>> x6, 
	tapa::mmap<bits<DataVec>> x7, 
	tapa::mmap<bits<DataVec>> y0, 
	tapa::mmap<bits<DataVec>> y1, 
	tapa::mmap<bits<DataVec>> y2, 
	tapa::mmap<bits<DataVec>> y3, 
	tapa::mmap<bits<DataVec>> y4, 
	tapa::mmap<bits<DataVec>> y5, 
	tapa::mmap<bits<DataVec>> y6, 
	tapa::mmap<bits<DataVec>> y7, 
	int poly_num){

	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramrd_streams("dramrd_streams");
	tapa::streams<Data, 2*BU*GROUP_CH_NUM, POLY_FIFO_DEPTH_M> dramwr_streams("dramwr_streams");
	tapa::streams<Data2, BU, 2> core_istreams("core_istreams");
	tapa::streams<Data2, BU, 2> core_ostreams("core_ostreams");

	tapa::task()
		.invoke<tapa::join>(read_dram_m, x0, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x1, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x2, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x3, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x4, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x5, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x6, dramrd_streams, poly_num)
		.invoke<tapa::join>(read_dram_m, x7, dramrd_streams, poly_num)
		.invoke<tapa::detach>(read_collect_m, dramrd_streams, core_istreams)

		.invoke<tapa::detach>(ntt_core, core_istreams, core_ostreams)

		.invoke<tapa::detach>(write_dist_m, dramwr_streams, core_ostreams)
		.invoke<tapa::join>(write_dram_m, y0, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y1, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y2, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y3, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y4, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y5, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y6, dramwr_streams, poly_num)
		.invoke<tapa::join>(write_dram_m, y7, dramwr_streams, poly_num)
	;
}

#endif //MCH

void ntt(tapa::mmaps<bits<DataVec>, 2*CH> hbm_ch, int poly_num){

	tapa::task()
#if GROUP_CH_NUM == 1
		.invoke<tapa::join, GROUP_NUM>(ntt_group_dram1, hbm_ch, hbm_ch, poly_num )
#elif GROUP_CH_NUM == 2
		.invoke<tapa::join, GROUP_NUM>(ntt_group_dram2, hbm_ch, hbm_ch, hbm_ch, hbm_ch, poly_num )
#elif GROUP_CH_NUM == 4
		.invoke<tapa::join, GROUP_NUM>(ntt_group_dram4, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, poly_num )
#elif GROUP_CH_NUM == 8
		.invoke<tapa::join, GROUP_NUM>(ntt_group_dram8, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, hbm_ch, poly_num )
#else
    #error "GROUP_CH_NUM must be one of 1, 2, 4, or 8"
#endif
	;
}

