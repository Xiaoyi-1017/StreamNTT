#include <iostream>
#include "ntt.h"

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


void bf_unit(const int stage, const int bf_id, tapa::istream<Data2>& input_stream, tapa::ostream<Data2>& output_stream)
{
	const int stage_shift = stage + 1;
	const int shift = num_l_stage - stage_shift;
	const ap_uint<logDEPTH> mask = (1<<shift) -1;

	// memory for entry with EVEN indices
	Data mem0[DEPTH/2]; 
	Data mem1[DEPTH/2];
#pragma HLS bind_storage variable=mem0 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem1 type=RAM_S2P impl=lutram 

	// memory for entry with ODD indices
	Data mem2[DEPTH/2]; 
	Data mem3[DEPTH/2];
#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=lutram 

	//memory read/write data count
	ap_uint<logDEPTH> read_idx = 0;     
	ap_uint<logDEPTH> write_idx = 0;     

	ap_uint<logDEPTH> read_limit = 0;     
	ap_uint<logDEPTH> write_limit = 0;     
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
		ap_uint<1> read_mem_idx = (read_idx >> shift) % 2;

		ap_uint<logDEPTH> read_upper_addr = (read_idx >> (shift+1)) << (shift);
		ap_uint<logDEPTH> read_lower_addr = (read_idx & mask);
		ap_uint<logDEPTH> raddr = read_upper_addr | read_lower_addr;

		ap_uint<1> write_mem_idx = (write_idx >> shift) % 2;

		ap_uint<logDEPTH> write_upper_addr = (write_idx >> (shift+1)) << (shift);
		ap_uint<logDEPTH> write_lower_addr = (write_idx & mask);
		ap_uint<logDEPTH> waddr = write_upper_addr | write_lower_addr;

		// Safety checks
		bool write_safe = write_limit != write_idx;
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

			int tw_idx = (((int)read_idx*BU) >> (logN - stage_shift)) + (1<<stage);

			butterfly(in_even, in_odd,  tw_factors[tw_idx], &out_even, &out_odd);

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

void l_stages(const int stage, tapa::istreams<Data2, BU>& input_stream, tapa::ostreams<Data2, BU>& output_stream){

	tapa::task()
		.invoke<tapa::detach, BU>(bf_unit, stage, tapa::seq(), input_stream, output_stream)
	;
}

void x_stages(tapa::istreams<Data2, BU>& input_streams, tapa::ostreams<Data2, BU>& output_streams){

	const int num_of_stages = logBU + 1;
	Data mem[num_of_stages+1][WIDTH];
	ap_uint<logDEPTH> i = 0;

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
#pragma HLS unroll 

				int current_stage = s + logN - logBU -1;
				int stage_shift = current_stage + 1;
				int stride = n >> ((current_stage) + 1);
				int next_stride = stride >> 1;

				// For division
				int shift = num_of_stages - (s + 1);

				// For remainder
				int mask = (1 << shift) -1;
				int next_mask = ( 1 << (shift-1) ) -1;

BUTTERFLY_LOOP:
				for(int idx = 0; idx < BU; idx++){
#pragma HLS unroll
					Data out_even, out_odd;

					int j = idx >> shift;
					int k = idx & mask;

					int tw_idx = ((i*BU+idx)>>(logN-stage_shift)) + (1<<current_stage);

					int ind_even = (next_stride == 0) ? ( j << (shift+1) ) 
						: ( j << (shift+1) ) + (k & next_mask) * 2 + (k >> (shift-1));
					int ind_odd = ind_even + stride;

					butterfly(mem[s][2*idx], mem[s][2*idx+1], tw_factors[tw_idx], &out_even, &out_odd);
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

			// For even inputs 0 ~ n/2-1
			if (write_idx % 2 == 0){ // 0, 512 - 2, 514 ...
				data_e = mem0[wr_s][write_idx/2];
				data_o = mem2[wr_s][write_idx/2];
			}
			else{            // 1, 513 - 3, 515 ...
				data_e = mem1[wr_s][write_idx/2];
				data_o = mem3[wr_s][write_idx/2];
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

/*
void input_mem_stage(tapa::istream<Data2>& i_stream, tapa::ostream<Data2>& o_stream){

	// memory for entry with EVEN indices
	Data mem0[DEPTH/2]; 
	Data mem1[DEPTH/2];
#pragma HLS bind_storage variable=mem0 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem1 type=RAM_S2P impl=lutram 

	// memory for entry with ODD indices
	Data mem2[DEPTH/2]; 
	Data mem3[DEPTH/2];
#pragma HLS bind_storage variable=mem2 type=RAM_S2P impl=lutram 
#pragma HLS bind_storage variable=mem3 type=RAM_S2P impl=lutram 

	//memory read/write data count
	ap_uint<logDEPTH> read_idx = 0;     
	ap_uint<logDEPTH> write_idx = 0;     

	ap_uint<logDEPTH> read_limit = 0;     
	ap_uint<logDEPTH> write_limit = 0;     

	bool set_read_limit = false;
	bool set_write_limit = false;
	bool mem_empty = true;


INPUT_MEM_STAGE_LOOP:
	for(;;){
#pragma HLS pipeline II = 1
#pragma HLS dependence variable=mem0 type=inter false
#pragma HLS dependence variable=mem1 type=inter false
#pragma HLS dependence variable=mem2 type=inter false
#pragma HLS dependence variable=mem3 type=inter false

		bool write_safe = write_limit != write_idx || mem_empty == true;
		bool read_safe = read_limit != read_idx;

		if( set_write_limit == true ){
			write_limit = read_idx>>1;	
		}
		if( set_read_limit == true ){
			read_limit = write_idx<<1;	
		}

		if( set_read_limit == true ){
			mem_empty = false;
		}
		else if( set_write_limit == true && write_idx == 0 && read_idx == 0 ){
			mem_empty = true;
		}
		set_write_limit = false;
		set_read_limit = false;

		if( read_safe == true ){
			Data data_e;
			Data data_o;

			ap_uint<logDEPTH> raddr = read_idx(logDEPTH - 1, 1);
     
			// For even inputs 0 ~ n/2-1
			if( read_idx[0] == 0){ // 0, 512 - 2, 514 ...
				data_e = mem0[raddr];
				data_o = mem2[raddr];
			}
			else{            // 1, 513 - 3, 515 ...
				data_e = mem1[raddr];
				data_o = mem3[raddr];
			}

			Data2 data = (data_o, data_e);
			o_stream.write(data);

			if( read_idx[0] == 1){    
				set_write_limit = true;
			}

			read_idx++;
		}

		if( write_safe == true && !i_stream.empty() ){
			Data2 data = i_stream.read();
			Data data_e = data(K-1,0);
			Data data_o = data(2*K-1,K);

			ap_uint<logDEPTH> waddr = write_idx(logDEPTH - 2, 0);     

			if( write_idx[logDEPTH - 1] == 0){
				// For even inputs 0 ~ n/2-1      
				mem0[waddr] = data_e;
				mem1[waddr] = data_o;
			}
			else{
				// For odd inputs n/2 ~ n-1
				mem2[waddr] = data_e;
				mem3[waddr] = data_o;
			}

			if( write_idx[logDEPTH - 1] == 1){    
				set_read_limit = true;
			}

			write_idx++;
		}
	}
}
*/

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
		.invoke<tapa::detach, num_l_stage>(l_stages, tapa::seq(), core_streams, core_streams)
		.invoke<tapa::detach>(x_stages, core_streams, core_ostreams); 
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

/*
void ntt(tapa::mmaps<bits<DataVec>, CH> x, tapa::mmaps<bits<DataVec>, CH> y, int poly_num){

#ifdef MCH
	tapa::streams<Data, 2*BU*CH, POLY_FIFO_DEPTH_M> dramrd_streams("dramrd_streams");
	tapa::streams<Data, 2*BU*CH, POLY_FIFO_DEPTH_M> dramwr_streams("dramwr_streams");
#else
	tapa::streams<DataVec, NUM_CORE, POLY_FIFO_DEPTH_S> dramrd_streams("dramrd_streams");
	tapa::streams<DataVec, NUM_CORE, POLY_FIFO_DEPTH_S> dramwr_streams("dramwr_streams");
#endif
	tapa::streams<Data, BU*NUM_CORE, 2> core_istreams_e("core_istreams_e");
	tapa::streams<Data, BU*NUM_CORE, 2> core_istreams_o("core_istreams_o");

	tapa::streams<Data, BU*NUM_CORE, 2> core_ostreams_e("core_ostreams_e");
	tapa::streams<Data, BU*NUM_CORE, 2> core_ostreams_o("core_ostreams_o");

	tapa::task()
#ifdef MCH
		.invoke<tapa::join, CH>(read_dram_m, x, dramrd_streams, poly_num)
		.invoke<tapa::detach, GROUP_NUM>(read_collect_m, dramrd_streams, core_istreams_e, core_istreams_o)
#else
		.invoke<tapa::join, CH>(read_dram_s, x, dramrd_streams, poly_num)
	 	.invoke<tapa::detach, NUM_CORE>(read_dist_s, dramrd_streams, core_istreams_e, core_istreams_o)
#endif

		.invoke<tapa::detach, NUM_CORE>(ntt_core, core_istreams_e, core_istreams_o, core_ostreams_e, core_ostreams_o)

#ifdef MCH
		.invoke<tapa::detach, GROUP_NUM>(write_dist_m, dramwr_streams, core_ostreams_e, core_ostreams_o)
		.invoke<tapa::join, CH>(write_dram_m, y, dramwr_streams, poly_num)
#else
		.invoke<tapa::detach, NUM_CORE>(write_reshape_s, dramwr_streams, core_ostreams_e, core_ostreams_o)
		.invoke<tapa::join, CH>(write_dram_s, y, dramwr_streams, poly_num)
#endif
	;

}
*/
