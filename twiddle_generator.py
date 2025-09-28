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
    X_BASE = []
    for s_x in range(num_x_stage):
        s_cur = s_x + num_l_stage
        shift  = logN - (s_cur + 1)
        row = []
        for lane in range(BU):
            tw_idx = (lane >> shift) + (1 << s_cur)
            row.append(int(tw_factors[tw_idx]))
        X_BASE.append(row)
    TWF_BASE = []
    for v in L_BASE:
        TWF_BASE.append([v] * BU)
    TWF_BASE.extend([list(r) for r in X_BASE])
    
    return TWF_BASE

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
