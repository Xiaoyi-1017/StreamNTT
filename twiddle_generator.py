import numpy as np
from sympy import primitive_root, mod_inverse

def reverse_bits(num, bits):
    """
    Reverse the bits of an integer `num` within the range defined by `bits`.
    """
    reversed_num = 0
    for i in range(bits):
        if num & (1 << i):
            reversed_num |= 1 << ((bits - 1) - i)
    return reversed_num

def bit_reverse_array(arr):
    """
    Perform bit-reversal ordering on the input array `arr`.
    """
    n = len(arr)
    bits = int(np.log2(n))
    out = np.zeros_like(arr)
    for i in range(n):
        out[reverse_bits(i, bits)] = arr[i]
    return out

def get_nth_root_of_unity(n, mod):
    """
    Calculate the nth root of unity (omega) modulo `mod` using sympy.
    """
    g = primitive_root(mod)  # Find a primitive root of the modulus
    omega = pow(g, (mod - 1) // n, mod)  # Calculate the nth root of unity
    return omega


def get_psi_original(n, q, omega):
    """
    Find a suitable psi value that satisfies the conditions.
    """
    for psi in range(q):
        mul = pow(psi, 2 * n, q)
        if mul == 1 and pow(psi, 2, q) == omega % q:
            return psi
    return None
import concurrent.futures

def check_psi_range(start, end, n, q, omega):
    """
    Check for a valid psi in the range [start, end).
    Returns psi if found, or None if not.
    """
    for psi in range(start, end):
        # Check the two conditions:
        # 1. psi^(2*n) mod q == 1
        # 2. psi^2 mod q == omega mod q
        if pow(psi, 2 * n, q) == 1 and pow(psi, 2, q) == (omega % q):
            return psi
    return None

def get_psi(n, q, omega, num_workers=8):
    """
    Find a suitable psi value in [0, q) that satisfies:
       psi^(2*n) mod q == 1
       psi^2 mod q == omega mod q
    This version splits the search across several processes.
    """
    # Determine a chunk size so that we roughly split the work equally.
    chunk_size = (q + num_workers - 1) // num_workers

    # Create a process pool.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each chunk.
        futures = []
        for start in range(0, q, chunk_size):
            end = min(start + chunk_size, q)
            futures.append(
                executor.submit(check_psi_range, start, end, n, q, omega)
            )

        # As soon as one task returns a non-None result, we can return it.
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                # Optionally, cancel the other tasks.
                for f in futures:
                    f.cancel()
                return result

    # If no psi was found, return None.
    return None

def get_nth_root_of_unity_and_psi(n, mod):
    """
    Calculate the nth root of unity (omega) modulo `mod` using sympy.
    """
    
    g = primitive_root(mod)  # Find a primitive root of the modulus

    
    omega = pow(g, (mod - 1) // n, mod)  # Calculate the nth root of unity

    # psi = get_psi(n, mod, omega)
    psi = get_psi_original(n, mod, omega)
    
    return omega, psi



def twiddle_generator_BR(mod, psi, n):
    """
    Generate bit-reversed twiddle factors.
    """
    arr = [1]*n

    for i in range(n-1):
        arr[i+1] = (arr[i]*psi) % mod

    return bit_reverse_array(arr)

def twiddle_base_generator(mod, psi, n, BU, logN, logBU):
    """
    Generate on-the-fly twiddle factors.
    """
    tw_factors = twiddle_generator_BR(mod, psi, n)
    
    # On-the-fly twf_base_table
    num_l_stage = logN - logBU - 1
    num_x_stage = logBU + 1
    L_BASE = [int(tw_factors[1 << s]) for s in range(num_l_stage)]
    TWF_X_BASE = []
    for s_x in range(num_x_stage):
        s_cur = s_x + num_l_stage
        shift  = logN - (s_cur + 1)
        row = []
        for lane in range(BU):
            tw_idx = (lane >> shift) + (1 << s_cur)
            row.append(int(tw_factors[tw_idx]))
        TWF_X_BASE.append(row)
 
    X_BASE = []
    for s_x, row in enumerate(TWF_X_BASE):
    	step = BU >> s_x
    	uniq = 1 << s_x # 1,2,4,...,BU
    	X_BASE.extend(row[0 : uniq * step : step])
	 
    return L_BASE, X_BASE
    
def twiddle_base_generator_lanes(mod, psi, n, BU, logN, logBU, K, TFG_II=2):
    """
    Generate on-the-fly twiddle bases for L-stage and X-stage.

    Args:
        mod, psi, n, BU, logN, logBU, K: usual parameters
        TFG_II: supported values = 1 / 2 / 3

    Returns:
        tw_l_base_lane0: length = num_l_stage
        tw_l_base_lane1: length = num_l_stage_ge1
        tw_l_base_lane2: length = num_l_stage_ge2
        tw_l_gamma:      length depends on TFG_II
        tw_x_base:       unchanged 1D flattened X-stage base table
        delta:           derived from mod and K
    """
    if TFG_II not in (1, 2, 3):
        raise ValueError(f"Unsupported TFG_II={TFG_II}, expected 1/2/3")

    tw_factors = twiddle_generator_BR(mod, psi, n)

    num_l_stage = logN - logBU - 1
    num_x_stage = logBU + 1

    num_l_stage_ge1 = num_l_stage - 1
    num_l_stage_ge2 = num_l_stage - 2
    num_l_stage_ge3 = num_l_stage - 3

    tw_l_base_lane0 = [int(tw_factors[1 << s]) for s in range(num_l_stage)]

    tw_l_base_lane1 = [
        (tw_l_base_lane0[stage + 1] * tw_l_base_lane0[stage]) % mod
        for stage in range(num_l_stage_ge1)
    ]

    tw_l_base_lane2 = [
        (tw_l_base_lane0[stage + 2] * tw_l_base_lane0[stage]) % mod
        for stage in range(num_l_stage_ge2)
    ]
    
    # lane3[stage] = lane0[stage+2] * lane0[stage+1] * lane0[stage] mod q
    tw_l_base_lane3 = [
        (tw_l_base_lane0[stage + 2] * tw_l_base_lane1[stage]) % mod
         for stage in range(1) # No need to be num_l_stage_ge3
    ]

    if TFG_II == 1:
        tw_l_gamma = [
            ((tw_l_base_lane0[stage + 1] << K) // mod)
            for stage in range(num_l_stage_ge2)
        ]
    elif TFG_II == 2:
        tw_l_gamma = [
            ((tw_l_base_lane0[stage] << K) // mod)
            for stage in range(num_l_stage_ge2)
        ]
    else:  # TFG_II == 3
        tw_l_gamma = [
            ((tw_l_base_lane1[stage + 1] << K) // mod)
            for stage in range(num_l_stage_ge3)
        ]

    TWF_X_BASE = []
    for s_x in range(num_x_stage):
        s_cur = s_x + num_l_stage
        shift = logN - (s_cur + 1)
        row = []
        for lane in range(BU):
            tw_idx = (lane >> shift) + (1 << s_cur)
            row.append(int(tw_factors[tw_idx]))
        TWF_X_BASE.append(row)

    tw_x_base = []
    for s_x, row in enumerate(TWF_X_BASE):
        step = BU >> s_x
        uniq = 1 << s_x
        tw_x_base.extend(row[0: uniq * step: step])

    delta = (1 << K) - mod
    if delta <= 0:
        raise ValueError(f"Invalid DELTA derived from mod={mod}, K={K}")

    return tw_l_base_lane0, tw_l_base_lane1, tw_l_base_lane2, tw_l_base_lane3, tw_l_gamma, tw_x_base, delta

def is_prime64(q) -> bool:
    """
    Deterministic Miller–Rabin for 64-bit integers.
    """
    if q < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    for p in small_primes:
        if q == p:
            return True
        if q % p == 0:
            return False
    d = q - 1
    r = 0
    while d & 1 == 0:
        d >>= 1
        r += 1
    
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % q == 0:
            continue
        x = pow(a, d, q)
        if x == 1 or x == q - 1:
            continue
        for _ in range(r-1):
            x = (x * x) % q
            if x == q - 1:
                break
        else:
            return False
    return True
    
def get_nth_root_of_unity_and_psi_fast(n, mod):
    """
    Used for prime Q and 2n | (Q-1); return omega, psi, where omega = psi^2
    """
    if (mod - 1) % (2 * n) != 0:
        raise ValueError("2*N does not divide Q-1")
    if not is_prime64(mod):
    	raise ValueError("Q is not priime")
    
    g = primitive_root(mod)
    omega = pow(g, (mod - 1) // n, mod)
    psi = pow(g, (mod - 1) // (2 * n), mod)
    print("Current psi is: ", psi)
    
    return omega, psi

def main():
    q_list = [12289, 8380417]
    
    q_dict = {}

    for q in q_list:
        n = 64
        q_dict[q] = {}

        while(n <= 1024):
            omega, psi = get_nth_root_of_unity_and_psi(n, q)

            print(f"{q}, {n}: {omega}, {psi}")

            q_dict[q][n] = psi

            n *= 2
            
    print(q_dict)

    ## 8380417: {64: 3241972, 128: 6644104, 256: 6458423, 512: 7829487, 1024: 7352248}

if __name__ == "__main__":
    main()

