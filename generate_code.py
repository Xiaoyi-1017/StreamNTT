#!/usr/bin/env python3
import argparse
import math
import os
import shutil
from twiddle_generator import get_nth_root_of_unity_and_psi, twiddle_base_generator

def check_q_and_data_length(q):
    
    q_list = {12289: 14, 7681: 13, 8380417: 23, 3221225473: 32}

    if q not in q_list:
        print(f"Current Q value is not supported")
        return
    
    K = q_list[q]
    
    # Setting up host_data format and bits
    bits = None
    data_format = None

    if K <= 16:
        data_format = "uint16_t"
        bits = 16
    elif K <= 32:
        data_format = "uint32_t"
        bits = 32
    elif K <= 64:
        data_format = "uint64_t"
        bits = 64
    else:
        print(f"Current log q is too large.")
        return
    
    return K, bits, data_format


def round_to_nearest_power_of_2(n):
    if n < 1:
        return 1

    power_low = 2 ** math.floor(math.log2(n))
    power_high = 2 ** math.ceil(math.log2(n))

    return power_low if (n - power_low) < (power_high - n) else power_high


def generate_ini(ch, folder):
    if not (1 <= ch <= 16):
        raise ValueError("CH must be between 1 and 16.")

    lines = ["[connectivity]"]
    lines_1 = ["[connectivity]"]

    for i in range(2*ch):
        lines.append(f"sp=ntt.hbm_ch_{i}:HBM[{i}]")
        lines_1.append(f"sp=ntt_1.hbm_ch_{i}:HBM[{i}]")

    filename = os.path.join(folder, "link_config.ini")
    with open(filename, "w") as file:
        file.write("\n".join(lines))

    #print(f"{filename} has been generated.")

    filename_1 = os.path.join(folder, "link_config_1.ini")
    with open(filename_1, "w") as file_1:
        file_1.write("\n".join(lines_1))


def generate_makefile(CH, GROUP_NUM, GROUP_CH_NUM, folder):
    template_file = "./templates/Makefile"
    with open(template_file, "r") as file:
        content = file.read()

    content = content.replace("{CH}", str(CH))
    content = content.replace("{GROUP_NUM}", str(GROUP_NUM))
    content = content.replace("{GROUP_CH_NUM}", str(GROUP_CH_NUM))

    output_file = os.path.join(folder, "Makefile")
    with open(output_file, "w") as file:
        file.write(content)

    #print(f"{output_file} has been generated.")


def generate_header(n, mod, K, bits, data_format, BU, CH, RATE, folder):
    WIDTH = 2*BU
    DEPTH = n / WIDTH
    logDEPTH = int(math.log2(DEPTH))

    logN = int(math.log2(n))
    logBU = int(math.log2(BU))

    DATA_BSIZE = int(bits/8)
    DataCHLen = int(64 / DATA_BSIZE)

    EffDataCHLen = round_to_nearest_power_of_2(DataCHLen*RATE)
    print(f"DataCHLen: {DataCHLen}, EffDataCHLen: {EffDataCHLen}")

    NUM_CORE = int(CH * EffDataCHLen / (2*BU))

    if NUM_CORE < 1:
        raise ValueError(f"Error: CH*{EffDataCHLen} must be equal to or larger than 2*BU.")

    GROUP_NUM = int(min(CH, NUM_CORE))
    GROUP_CH_NUM = int(CH / GROUP_NUM)
    MCH = int(GROUP_CH_NUM > 1)
    print(f"NUM_CORE: {NUM_CORE}, GROUP_NUM: {GROUP_NUM}, GROUP_CH_NUM: {GROUP_CH_NUM}")

    # it takes too much to calculate psi for large q (3221225473)
    psi = None
    psi_dict = {
                7681: {64: 3449, 128: 2028, 256: 535},
                12289: {64: 140, 128: 8340, 256: 3400, 512: 1987, 1024: 1945}, 
                8380417: {64: 3241972, 128: 1736313, 256: 1921994, 512: 550930, 1024: 1028169},
                3221225473: {64: 1292405718, 128: 1262731197, 256 : 764652596, 512 : 1365964089, 1024: 1168849724, 65536: 1800384970, 131072: 1358427440}
               }
    
    if mod in psi_dict:
        if n in psi_dict[mod]:
            psi = psi_dict[mod][n]
        else:
            omega, psi = get_nth_root_of_unity_and_psi(n, mod)

    else:
        print("Current Modulo is not supported: ", mod)
        return

    tw_l_base, tw_x_base = twiddle_base_generator(mod, psi, n, BU, logN, logBU)


    # template_file = f"./{folder}/src/ntt.h"
    target_file = os.path.join(folder, "src/ntt.h")

    with open(target_file, "r") as file:
        header_content = file.read()
    
    header_content = header_content.replace("{K}", str(K))
    header_content = header_content.replace("{DATA_FORMAT}", data_format)
    header_content = header_content.replace("{MOD}", str(mod))
    header_content = header_content.replace("{N}", str(n))
    header_content = header_content.replace("{logN}", str(logN))
    header_content = header_content.replace("{BU}", str(BU))
    header_content = header_content.replace("{logBU}", str(logBU))
    header_content = header_content.replace("{logDEPTH}", str(logDEPTH))
    header_content = header_content.replace("{CH}", str(CH))
    header_content = header_content.replace("{MCH}", str(MCH))
    header_content = header_content.replace("{EffDataCHLen}", str(EffDataCHLen))
    header_content = header_content.replace("{DATA_BSIZE}", str(DATA_BSIZE))
    header_content = header_content.replace("{GROUP_NUM}", str(GROUP_NUM))
    header_content = header_content.replace("{GROUP_CH_NUM}", str(GROUP_CH_NUM))
    header_content = header_content.replace("{TWF_L_BASE}", "const Data tw_l_base[num_l_stage] = {" + \
    						", ".join(str(int(x)) for x in tw_l_base) + "};")
    header_content = header_content.replace("{TWF_X_BASE}", "const Data tw_x_base[tw_x_base_size] = {" + \
    						", ".join(str(int(x)) for x in tw_x_base) + "};")
    header_content = header_content.replace("{PSI}", str(psi))

    # output_file = os.path.join(folder, "./src/ntt.h")
    with open(target_file, "w") as file:
        file.write(header_content)
            
    #print(f"{target_file} has been generated.")

    generate_makefile(CH, GROUP_NUM, GROUP_CH_NUM, folder)


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of NTT cores.")
    parser.add_argument("-N", type=int, default=1024, help="The size of N.")
    parser.add_argument("-q", type=int, default=3221225473, help="Declare appropriate q")
    # parser.add_argument("-bits", type=int, default=32, help="Coefficient bit-width")
    parser.add_argument("-BU", type=int, default=8, help="The size of BU.")
    parser.add_argument("-CH", type=int, default=16, help="The number of memory channels (input) ")
    parser.add_argument("-RATE", type=float, default=0.5, help="Effective DRAM transfer rate (0 - 1.0)")

    args = parser.parse_args()

    N = args.N
    mod = args.q
    # bits = args.bits
    BU = args.BU
    CH = args.CH
    RATE = args.RATE
    
    K, bits, data_format = check_q_and_data_length(mod)    

    veclen = 512 // bits

    if (CH*veclen) % (2*BU) != 0:
        print(f"{CH} * {veclen} should be divisible by {2 * BU}. Design not feasible.")
        return

    print(f"Values used -> N: {N}, q: {mod}, HostData: {data_format}, BU: {BU}, CH: {CH}, RATE: {RATE}, veclen: {veclen}")

    # Create new folder
    folder_name = f"N{N}_BU{BU}_CH{CH}_q{mod}"
    if os.path.exists(folder_name):
        print(f"Folder exists: {folder_name}. No folder created.")
    else:
        print(f"Creating a new folder: {folder_name}")
        os.makedirs(folder_name)

        # Copy all existing files to the new folder
        source_folder = "./templates"
        if os.path.exists(source_folder):
            for item in os.listdir(source_folder):
                s = os.path.join(source_folder, item)
                d = os.path.join(folder_name, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

        # Generate new files in the folder
        generate_ini(CH, folder_name)
        generate_header(N, mod, K, bits, data_format, BU, CH, RATE, folder_name)

 
if __name__ == "__main__":
    main()
