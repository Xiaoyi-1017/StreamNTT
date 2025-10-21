#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "ntt.h"

int bit_reverse(int i){
	ap_uint<logN> x = (ap_uint<logN>) i;
	ap_uint<logN> reversed_idx = x.reverse();
	return (int) reversed_idx;
}

/* Changed to prevent overflow */
HostData mod_power(HostData x, int exp, HostData mod){
	unsigned __int128 result = 1;
	for(int i = 0; i < exp; ++i){
		result = (result * (unsigned __int128)x) % mod;
	}

	HostData result_output = static_cast<HostData>(result);

	return result_output;
}

/* Changed to prevent overflow */

void sw_ntt(std::vector<HostData, tapa::aligned_allocator<HostData>> A, std::vector<HostData, tapa::aligned_allocator<HostData>> &out_sw, 
		HostData psi, HostData q, const int POLY_NUM){
    
    // Precompute powers of omega_n = psi^2
    HostData omega_n = (HostData) mod_power(psi, 2, q);
    std::vector<HostData> omega_powers(n/2);
    omega_powers[0] = 1;
    for (int i = 1; i < n/2; ++i) {
        omega_powers[i] = static_cast<HostData>((unsigned __int128)omega_powers[i-1] * omega_n % q);
    }

#pragma omp parallel for
    for (int p = 0; p < POLY_NUM; ++p) {
        std::vector<HostData> data(n);
        
        // Copy input and apply twiddle factors
        for (int j = 0; j < n; ++j) {
            HostData psi_j = (HostData) mod_power(psi, j, q);
            data[j] = static_cast<HostData>((unsigned __int128)A[p*n + j] * psi_j % q);
        }
        
        // Bit-reverse
        std::vector<HostData> temp(n);
        for (int i = 0; i < n; ++i) {
            temp[i] = data[bit_reverse(i)];
        }
        data.swap(temp);
        
        // Cooley-Tukey DIT FFT
        for (int stage = 0; stage < logN; ++stage) {
            int m = 1 << (stage + 1);  // Current block size
            int half = m >> 1;          // Half block size
            int stride = n / m;         // Stride for omega powers
            
            for (int k = 0; k < n; k += m) {
                for (int j = 0; j < half; ++j) {
                    HostData w = omega_powers[j * stride];
                    
                    int i1 = k + j;
                    int i2 = i1 + half;
                    
                    HostData u = data[i1];
                    HostData v = data[i2];
                    
                    unsigned __int128 wv128 = (unsigned __int128)w * (unsigned __int128)v;
                    HostData wv = static_cast<HostData>(wv128 % q);
                    data[i1] = (u + wv) % q;
                    data[i2] = static_cast<HostData>(((unsigned __int128)u + q - wv) % q);
                }
            }
        }
        
        // Copy result
        for (int i = 0; i < n; ++i) {
            out_sw[p*n + i] = data[i];
        }
    }
}

void bit_reverse_hw_out(std::vector<HostData, tapa::aligned_allocator<HostData>> out_hw, std::vector<HostData, tapa::aligned_allocator<HostData>> &out_hw_BR, const int POLY_NUM){

#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		if( tid == 0 ){
			int nthreads = omp_get_num_threads();
		}
	}

#pragma omp parallel for
	//Bit-reversing the HW output
	for(int i = 0; i < POLY_NUM; i++){
		for(int j = 0; j < n; j++){
			out_hw_BR[n*i + bit_reverse(j)] = out_hw[n*i + j];
		}
	}
}

void copy_input(std::vector<HostData, tapa::aligned_allocator<HostData>> input, 
		std::vector<std::vector<HostData, tapa::aligned_allocator<HostData> >, std::allocator<std::vector<HostData, tapa::aligned_allocator<HostData> > >> &HBM_CH, int POLY_NUM){

	for(int i = 0; i < POLY_NUM; i++){
		int index = (((i/GROUP_CH_NUM)*(2*GROUP_CH_NUM))%(2*CH)) + (i%GROUP_CH_NUM); 
		for(int j = 0; j < n; j++){
			//X[index][(i/CH)*n+j] = rand() % mod;
			HBM_CH[index][(i/CH)*n+j] = input[i*n + j];
		}
	}
}

void copy_output(std::vector<std::vector<HostData, tapa::aligned_allocator<HostData> >, std::allocator<std::vector<HostData, tapa::aligned_allocator<HostData> > >> HBM_CH, 
		std::vector<HostData, tapa::aligned_allocator<HostData>> &out_hw, int POLY_NUM){

	for(int i = 0; i < POLY_NUM; i++){
		int index = (((i/GROUP_CH_NUM)*(2*GROUP_CH_NUM))%(2*CH)) + (i%GROUP_CH_NUM) + GROUP_CH_NUM; 
		for(int j = 0; j < n; j++){
			out_hw[i*n + j] = HBM_CH[index][(i/CH)*n+j];
		}
	}
}

/* Easier to match new hw outputs */

void anti_bit_reverse_sw_out(std::vector<HostData, tapa::aligned_allocator<HostData>>& sw_br, std::vector<HostData, tapa::aligned_allocator<HostData>>& sw_nat, int POLY_NUM){

	for (int p = 0; p < POLY_NUM; ++p) {
	    	int base = p * n;
	    	for (int j = 0; j < n; ++j) {
		      	int jr = bit_reverse(j);
		      	sw_nat[base + j] = sw_br[base + jr];
    		}
  	}
}

void ntt(tapa::mmaps<bits<DataVec>, 2*CH> hbm_ch, int poly_num);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {

	gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

	const int POLY_NUM = argc > 1 ? atoll(argv[1]) : 1000;

	const HostData mod = MOD;

	const int N = n;

	const int ADJ_FACTOR = CH > NUM_CORE ? CH : NUM_CORE;
	const int ADJ_POLY_NUM = ((POLY_NUM - 1) / ADJ_FACTOR + 1) * ADJ_FACTOR;

	std::vector<HostData, tapa::aligned_allocator<HostData>> input(n*ADJ_POLY_NUM);

	std::vector<HostData, tapa::aligned_allocator<HostData>> out_sw(n*ADJ_POLY_NUM);
	std::vector<HostData, tapa::aligned_allocator<HostData>> out_hw(n*ADJ_POLY_NUM);
	std::vector<HostData, tapa::aligned_allocator<HostData>> out_hw_BR(n*ADJ_POLY_NUM);

	std::vector<std::vector<HostData, tapa::aligned_allocator<HostData>>> HBM_CH(2*CH);

	int last_stride = n >> (logN - logBU -1);

	// Generate twiddle factors
	std::cout << "n: " << n << " mod: " << mod << std::endl;
	std::cout << "Number of polynomials: " << ADJ_POLY_NUM << " (adjusted from " << POLY_NUM << ")" << std::endl; 
	std::cout << "Number of NTT cores: " << NUM_CORE << std::endl;
	std::cout << "Number of butterfly units per NTT core per stage: " << BU << std::endl; 
	std::cout << "Number of total butterfly units: " << BU*NUM_CORE*logN << std::endl; 
	std::cout << "Number of input and output DRAM channels: " << CH << std::endl;
	std::cout << "DEPTH per polynomial: " << DEPTH << std::endl;

	srand(time(NULL));

	// Create the test HostData 
	for(int i = 0; i < ADJ_POLY_NUM; i++){
		for(int j = 0; j < n; j++){
			//   input[i*n + j] = rand() % mod;
			input[i*n + j] = (i*n + j) % mod;
		}
	}

	// Resize each channel into appropriate size
	int CHANNEL_SIZE = n*ADJ_POLY_NUM/CH;

	if( CHANNEL_SIZE * sizeof(HostData) > 256 * 1024 * 1024 ){
		std::cout << "Error: HostData size of each channel (" << CHANNEL_SIZE * sizeof(HostData) << ") exceeds 256 MB" << std::endl;
		exit(1);
	}  

	for (int cc = 0; cc < 2*CH; ++cc) {
		HBM_CH[cc].resize(CHANNEL_SIZE, 0);
	}

	copy_input(input, HBM_CH, ADJ_POLY_NUM);

	std::cout << "Test HostData generation DONE" << std::endl;

	sw_ntt(input, out_sw, psi, mod, ADJ_POLY_NUM);

	std::cout << "SW Computation Done. Launching FPGA Kernel..." << std::endl;

	int64_t kernel_time_ns =
		tapa::invoke(ntt, FLAGS_bitstream,
				tapa::read_write_mmaps<HostData, 2*CH>(HBM_CH).reinterpret<bits<DataVec>>(),
				ADJ_POLY_NUM);


	//std::cout << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
	std::cout << "Kernel time: " << (double)kernel_time_ns * 1e-6  << " ms" << std::endl;
	//std::cout << "kernel time: " << kernel_time_ns * 1e-3 << " us" << std::endl;

	std::cout << "Kernel throughput: " << (double)ADJ_POLY_NUM / kernel_time_ns * 1e3 << " M Polynomials/s" << std::endl; 
	std::cout << "Eff BW per HBM channel: " << (double)CHANNEL_SIZE*sizeof(HostData) / kernel_time_ns << " GB/s" << std::endl; 

	std::cout << "Copying output ...\n";

	copy_output(HBM_CH, out_hw, ADJ_POLY_NUM);    

	// bit_reverse_hw_out(out_hw, out_hw_BR, ADJ_POLY_NUM);
	std::vector<HostData, tapa::aligned_allocator<HostData>> out_sw_nat(n*ADJ_POLY_NUM);
	anti_bit_reverse_sw_out(out_sw, out_sw_nat, ADJ_POLY_NUM);

	std::cout << "Done\n";

	// Compare the results of the Device to the simulation

	int err_cnt = 0;
	auto bitrev_k = [&](int x){
		unsigned r=0; for(unsigned i=0;i<logDEPTH;++i) r=(r<<1)|((x>>i)&1u);
		return (int)r;
	};

	for(int i = 0; i<ADJ_POLY_NUM; i++){
		for (int g = 0; g < DEPTH; ++g) { // natural group idx
	    		const int gr = bitrev_k(g); // HW group idx
			for (int t = 0; t < WIDTH; ++t) { // intra-group offset
				const int j_sw = i*n + g*WIDTH + t; // natural layout
				const int j_hw = i*n + gr*WIDTH + t; // HW layout
				if (out_sw_nat[j_sw] != out_hw[j_hw]) {
					err_cnt++;
					if(err_cnt < 32) printf("Error poly %d [g=%d,gr=%d,t=%d] j_sw=%d j_hw=%d sw:%" PRIu64 " hw:%" PRIu64 "\n",
					 i, g, gr, t, j_sw, j_hw,
					 (uint64_t)out_sw_nat[j_sw], (uint64_t)out_hw[j_hw]);
				}
			}
		}
	}

	if(err_cnt != 0){
		printf("FAILED! Error count : %d / %d\n", err_cnt, ADJ_POLY_NUM*n);
	}
	else{
		printf("PASSED!\n");
	}

	return EXIT_SUCCESS;
}
