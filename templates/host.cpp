#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>
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
void sw_ntt(const std::vector<HostData, tapa::aligned_allocator<HostData>> &A, std::vector<HostData, tapa::aligned_allocator<HostData>> &out_sw, HostData psi_val, HostData q, int poly_start, int poly_end){

    // Precompute powers of omega_n = psi^2
    HostData omega_n = (HostData) mod_power(psi_val, 2, q);
    std::vector<HostData> omega_powers(n/2);
    omega_powers[0] = 1;
    for (int i = 1; i < n/2; ++i) {
        omega_powers[i] = static_cast<HostData>((unsigned __int128)omega_powers[i-1] * omega_n % q);
    }

    for (int p = poly_start; p < poly_end; ++p) {
    	std::vector<HostData> data(n);

        // Copy input and apply twiddle factors.
        // psi_j accumulates psi_val^j (mod q) in O(1) per step -> O(n) total;
        // bit-exact vs the previous mod_power(psi_val, j, q) which was O(n^2).
        HostData psi_j = 1;
        for (int j = 0; j < n; ++j) {
        	data[j] = static_cast<HostData>((unsigned __int128)A[p*n + j] * psi_j % q);
        	psi_j = static_cast<HostData>((unsigned __int128)psi_j * psi_val % q);
        }

        // Bit-reverse
        std::vector<HostData> temp(n);
        for (int i = 0; i < n; ++i) {
        	temp[i] = data[bit_reverse(i)];
        }
        data.swap(temp);

        // Cooley-Tukey DIT FFT
        for (int stage = 0; stage < logN; ++stage) {
        	int m = 1 << (stage + 1);
        	int half = m >> 1;
       		int stride = n / m;

            for (int k = 0; k < n; k += m) {
                for (int j = 0; j < half; ++j) {
                	HostData w = omega_powers[j * stride];

                	int i1 = k + j;
                	int i2 = i1 + half;

                	HostData u = data[i1];
                	HostData v = data[i2];

                	unsigned __int128 wv128 = (unsigned __int128)w * (unsigned __int128)v;
                	HostData wv = static_cast<HostData>(wv128 % q);

                	unsigned __int128 sum = (unsigned __int128)u + wv;
                	data[i1] = (sum >= q) ? (HostData)(sum - q) : (HostData)sum;

                	unsigned __int128 diff = (unsigned __int128)u + q - wv;
                	data[i2] = (diff >= q) ? (HostData)(diff - q) : (HostData)diff;
                }
            }
        }

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

// Multi-core multi-prime routing assigns complete NUM_PRIMES cycles to each group.
// ADJ_POLY_NUM is rounded to a multiple of GROUP_NUM * NUM_PRIMES.
// GROUP_NUM == 1 degenerates to the original single-core layout.
void copy_input(std::vector<HostData, tapa::aligned_allocator<HostData>> input,
		std::vector<std::vector<HostData, tapa::aligned_allocator<HostData> >, std::allocator<std::vector<HostData, tapa::aligned_allocator<HostData> > >> &HBM_CH, int POLY_NUM){

	for(int i = 0; i < POLY_NUM; i++){
		int cycle_idx = i / NUM_PRIMES;
		int pos_in_cycle = i % NUM_PRIMES;
		int g = cycle_idx % GROUP_NUM;
		int within_group_cyc = cycle_idx / GROUP_NUM;
		int wgp = within_group_cyc * NUM_PRIMES + pos_in_cycle;
		int index = g * (2 * GROUP_CH_NUM) + (wgp % GROUP_CH_NUM);
		int offset = (wgp / GROUP_CH_NUM) * n;
		for(int j = 0; j < n; j++){
			HBM_CH[index][offset + j] = input[i*n + j];
		}
	}
}

static int bitrev_depth_m1(int x) {
	int r = 0;
	for (int i = 0; i < logDEPTH - 1; ++i)
		r = (r << 1) | ((x >> i) & 1);
	return r;
}

void copy_input_prepermute(std::vector<HostData, tapa::aligned_allocator<HostData>> input, std::vector<std::vector<HostData, tapa::aligned_allocator<HostData> >, std::allocator<std::vector<HostData, tapa::aligned_allocator<HostData> > >> &HBM_CH, int POLY_NUM){

	for(int i = 0; i < POLY_NUM; i++){
		int cycle_idx = i / NUM_PRIMES;
		int pos_in_cycle = i % NUM_PRIMES;
		int g = cycle_idx % GROUP_NUM;
		int within_group_cyc = cycle_idx / GROUP_NUM;
		int wgp = within_group_cyc * NUM_PRIMES + pos_in_cycle;
		int index = g * (2 * GROUP_CH_NUM) + (wgp % GROUP_CH_NUM);
		int offset = (wgp / GROUP_CH_NUM) * n;
		for(int j = 0; j < n; j++){
			int t = j / WIDTH;
			int s = j % WIDTH;
			int d = s % BU;
			int src;
			if (s < BU) {
				if (t < DEPTH/2)
					src = WIDTH * bitrev_depth_m1(t) + d;
				else
					src = WIDTH * bitrev_depth_m1(t - DEPTH/2) + BU + d;
			} else {
				if (t < DEPTH/2)
					src = n/2 + WIDTH * bitrev_depth_m1(t) + d;
				else
					src = n/2 + WIDTH * bitrev_depth_m1(t - DEPTH/2) + BU + d;
			}
			HBM_CH[index][offset + j] = input[i*n + src];
		}
	}
}

void copy_output(std::vector<std::vector<HostData, tapa::aligned_allocator<HostData> >, std::allocator<std::vector<HostData, tapa::aligned_allocator<HostData> > >> HBM_CH, std::vector<HostData, tapa::aligned_allocator<HostData>> &out_hw, int POLY_NUM){

	for(int i = 0; i < POLY_NUM; i++){
		int cycle_idx = i / NUM_PRIMES;
		int pos_in_cycle = i % NUM_PRIMES;
		int g = cycle_idx % GROUP_NUM;
		int within_group_cyc = cycle_idx / GROUP_NUM;
		int wgp = within_group_cyc * NUM_PRIMES + pos_in_cycle;
		int index = g * (2 * GROUP_CH_NUM) + GROUP_CH_NUM + (wgp % GROUP_CH_NUM);
		int offset = (wgp / GROUP_CH_NUM) * n;
		for(int j = 0; j < n; j++){
			out_hw[i*n + j] = HBM_CH[index][offset + j];
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

	const int N = n;

	// ADJ_POLY_NUM must satisfy BOTH:
	//   (a) divisible by GROUP_NUM * NUM_PRIMES (cyclic prime scheduling
	//       — every group sees a complete prime cycle).
	//   (b) divisible by max(CH, NUM_CORE) (channel/core distribution —
	//       copy_input / copy_output assigns whole polynomials per
	//       channel and per core).
	// The previous rule used max(...) which did not satisfy both when
	// the two factors are coprime (e.g. NUM_PRIMES=5, CH=4 → max=5 but
	// 5 % 4 != 0). The correct alignment is the LCM of the two factors.
	const int ADJ_FACTOR_CH    = CH > NUM_CORE ? CH : NUM_CORE;
	const int ADJ_FACTOR_MP    = GROUP_NUM * NUM_PRIMES;
	const int ADJ_FACTOR_FINAL =
		ADJ_FACTOR_CH / std::gcd(ADJ_FACTOR_CH, ADJ_FACTOR_MP) * ADJ_FACTOR_MP;
	const int ADJ_POLY_NUM =
		((POLY_NUM - 1) / ADJ_FACTOR_FINAL + 1) * ADJ_FACTOR_FINAL;

	std::cout << "======== Multi-Prime NTT CS1 (cyclic, " << NUM_PRIMES << " primes) ========" << std::endl;
	for (int p = 0; p < NUM_PRIMES; ++p) {
		std::cout << "q" << p << " = " << (uint64_t)MODS[p] << std::endl;
	}

	std::vector<HostData, tapa::aligned_allocator<HostData>> input(n*ADJ_POLY_NUM);

	std::vector<HostData, tapa::aligned_allocator<HostData>> out_sw(n*ADJ_POLY_NUM);
	std::vector<HostData, tapa::aligned_allocator<HostData>> out_hw(n*ADJ_POLY_NUM);

	std::vector<std::vector<HostData, tapa::aligned_allocator<HostData>>> HBM_CH(2*CH);

	std::cout << "n: " << n << std::endl;
	std::cout << "Number of polynomials: " << ADJ_POLY_NUM << " (adjusted from " << POLY_NUM << ")" << std::endl;
	std::cout << "Number of NTT cores: " << NUM_CORE << std::endl;
	std::cout << "Number of butterfly units per NTT core per stage: " << BU << std::endl;
	std::cout << "Number of total butterfly units: " << BU*NUM_CORE*logN << std::endl;
	std::cout << "Number of input and output DRAM channels: " << CH << std::endl;
	std::cout << "DEPTH per polynomial: " << DEPTH << std::endl;

	srand(time(NULL));

	// Create test data: use smallest prime range for all inputs
	HostData mod_min = (HostData)MODS[0];
	for (int p = 1; p < NUM_PRIMES; ++p) {
		HostData q_p = (HostData)MODS[p];
		if (q_p < mod_min) mod_min = q_p;
	}
	for(int i = 0; i < ADJ_POLY_NUM; i++){
		for(int j = 0; j < n; j++){
			input[i*n + j] = (i*n + j) % mod_min;
		}
	}

	// Resize each channel
	int CHANNEL_SIZE = n*ADJ_POLY_NUM/CH;

	if( CHANNEL_SIZE * sizeof(HostData) > 256 * 1024 * 1024 ){
		std::cout << "Error: HostData size of each channel (" << CHANNEL_SIZE * sizeof(HostData) << ") exceeds 256 MB" << std::endl;
		exit(1);
	}

	for (int cc = 0; cc < 2*CH; ++cc) {
		HBM_CH[cc].resize(CHANNEL_SIZE, 0);
	}

#if HOST_PREPERMUTE_INPUT
	auto t0_pp = std::chrono::high_resolution_clock::now();
	copy_input_prepermute(input, HBM_CH, ADJ_POLY_NUM);
	auto t1_pp = std::chrono::high_resolution_clock::now();
	std::cout << "Host input layout time: "
	          << std::chrono::duration<double, std::milli>(t1_pp - t0_pp).count()
	          << " ms" << std::endl;
#else
	copy_input(input, HBM_CH, ADJ_POLY_NUM);
#endif

	std::cout << "Test HostData generation DONE" << std::endl;

	// SW reference: cyclic per-polynomial prime assignment
	// Poly p uses mod_id = p % NUM_PRIMES
	// Parallelize across polynomials: each poly p writes a disjoint out_sw[p*n .. p*n+n) region,
	// input/PSIS/MODS are read-only, and sw_ntt keeps no shared mutable state (A is const&, omega_powers
	// is per-call). Poly p uses cyclic prime mid = p % NUM_PRIMES -- the SAME mapping the verifier uses.
	// OpenMP moved here from inside sw_ntt, where poly_end-poly_start==1 gave no useful parallelism.
#pragma omp parallel for schedule(dynamic)
	for (int p = 0; p < ADJ_POLY_NUM; ++p) {
		int mid = p % NUM_PRIMES;
		sw_ntt(input, out_sw, PSIS[mid], (HostData)MODS[mid], p, p+1);
	}

	std::cout << "SW Computation Done. Launching FPGA Kernel..." << std::endl;

	int64_t kernel_time_ns =
		tapa::invoke(ntt, FLAGS_bitstream,
				tapa::read_write_mmaps<HostData, 2*CH>(HBM_CH).reinterpret<bits<DataVec>>(),
				ADJ_POLY_NUM);

	std::cout << "Kernel time: " << (double)kernel_time_ns * 1e-6  << " ms" << std::endl;
	std::cout << "Kernel throughput: " << (double)ADJ_POLY_NUM / kernel_time_ns * 1e3 << " M Polynomials/s" << std::endl;
	std::cout << "Eff BW per HBM channel: " << (double)CHANNEL_SIZE*sizeof(HostData) / kernel_time_ns << " GB/s" << std::endl;

	std::cout << "Copying output ...\n";

	copy_output(HBM_CH, out_hw, ADJ_POLY_NUM);

	std::vector<HostData, tapa::aligned_allocator<HostData>> out_sw_nat(n*ADJ_POLY_NUM);
	anti_bit_reverse_sw_out(out_sw, out_sw_nat, ADJ_POLY_NUM);

	std::cout << "Done\n";

	// Compare: per-prime error reporting
	int err_cnt = 0;
	int err_cnt_per_prime[NUM_PRIMES] = {};
	auto bitrev_k = [&](int x){
		unsigned r=0; for(unsigned i=0;i<logDEPTH;++i) r=(r<<1)|((x>>i)&1u);
		return (int)r;
	};

	for(int i = 0; i<ADJ_POLY_NUM; i++){
		int mid = i % NUM_PRIMES;
		for (int g = 0; g < DEPTH; ++g) {
	    		const int gr = bitrev_k(g);
			for (int t = 0; t < WIDTH; ++t) {
				const int j_sw = i*n + g*WIDTH + t;
				const int j_hw = i*n + gr*WIDTH + t;
				if (out_sw_nat[j_sw] != out_hw[j_hw]) {
					err_cnt++;
					err_cnt_per_prime[mid]++;
					if(err_cnt < 32) printf("Error poly %d (mod_id=%d) [g=%d,gr=%d,t=%d] j_sw=%d j_hw=%d sw:%" PRIu64 " hw:%" PRIu64 "\n",
					 i, mid, g, gr, t, j_sw, j_hw,
					 (uint64_t)out_sw_nat[j_sw], (uint64_t)out_hw[j_hw]);
				}
			}
		}
	}

	if(err_cnt != 0){
		printf("FAILED! Error count : %d / %d", err_cnt, ADJ_POLY_NUM*n);
		for (int p = 0; p < NUM_PRIMES; ++p)
			printf(" (q%d: %d)", p, err_cnt_per_prime[p]);
		printf("\n");
		// Diagnostic: print which groups are correct for poly 0
		printf("Poly 0 correct groups (g/gr): ");
		for (int g = 0; g < DEPTH; ++g) {
			const int gr = bitrev_k(g);
			bool group_ok = true;
			for (int t = 0; t < WIDTH; ++t) {
				if (out_sw_nat[g*WIDTH + t] != out_hw[gr*WIDTH + t]) { group_ok = false; break; }
			}
			if (group_ok) printf("%d/%d ", g, gr);
		}
		printf("\n");
	}
	else{
		printf("PASSED! (%d polys, cyclic %d primes)\n", ADJ_POLY_NUM, NUM_PRIMES);
	}

	return EXIT_SUCCESS;
}

