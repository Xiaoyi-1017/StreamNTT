"""Generate rapidstream configuration files."""
__copyright__ = """
Copyright (c) 2024 RapidStream Design Automation, Inc. and contributors.
All rights reserved. The contributor(s) of this file has/have agreed to the
RapidStream Contributor License Agreement.
"""
import os
from pathlib import Path
import argparse


from rapidstream import get_u55c_vitis_device_factory, RapidStreamTAPA
from rapidstream.assets.floorplan.floorplan_config import FloorplanConfig
from rapidstream import PipelineConfig

CONFIG_DIR = Path("config_build")
FLOOR_PLAN_CONFIG = CONFIG_DIR / "floorplan_config.json"
DEVICE_CONFIG = CONFIG_DIR / "device_config.json"
PIPELINE_CONFIG = CONFIG_DIR / "pipeline_config.json"
VITIS_PLATFORM = "xilinx_u55c_gen3x16_xdma_3_202210_1"

def gen_config(ch: int, group_num: int, group_ch_num: int) -> None:
    """Generate configuration files."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    port_pre_assignments = {
        "ap_clk": "SLOT_X0Y0:SLOT_X0Y0",
        "ap_rst_n": "SLOT_X0Y0:SLOT_X0Y0",
        "interrupt": "SLOT_X0Y0:SLOT_X0Y0",
        "s_axi_control_.*": "SLOT_X0Y0:SLOT_X0Y0",
    }
    for i in range(2*ch):
        if i < 16:
            port_pre_assignments[f".*m_axi_hbm_ch_{i}_.*"] = "SLOT_X0Y0:SLOT_X0Y0"
        else:
            port_pre_assignments[f".*m_axi_hbm_ch_{i}_.*"] = "SLOT_X1Y0:SLOT_X1Y0"


    cell_pre_assignments = {}
    
    def assign_dram(gc, g, slot):
        prefix = "m" if group_ch_num > 1 else "s"
        cell_pre_assignments[f"ntt_group_dram{group_ch_num}_{g}/read_dram_{prefix}_{gc}"] = f"{slot}:{slot}"
        cell_pre_assignments[f"ntt_group_dram{group_ch_num}_{g}/write_dram_{prefix}_{gc}"] = f"{slot}:{slot}"

    for g in range(group_num): 
        c = g * 2 * group_ch_num 
        slot = "SLOT_X0Y0" if c < 16 else "SLOT_X1Y0"
        for gc in range(group_ch_num):
            assign_dram(gc, g, slot)


    floorplan_config = FloorplanConfig(
        port_pre_assignments=port_pre_assignments,
        cell_pre_assignments=cell_pre_assignments,
        max_seconds=1000,
        dse_range_min=0.7,
        dse_range_max=0.8,
    )
    floorplan_config.save_to_file(FLOOR_PLAN_CONFIG)

    left_ch = 2*ch if 2*ch < 16 else 16
    right_ch = 0 if 2*ch < 16 else 2*ch - left_ch 

    factory = get_u55c_vitis_device_factory(VITIS_PLATFORM)
    #Reserving LUTs/FFs for HBM memory sub-system
    factory.reduce_slot_area(0, 0, lut=5000*left_ch, ff=6500*left_ch)
    factory.reduce_slot_area(1, 0, lut=5000*right_ch+10000, ff=6500*right_ch)
    #Excluding DSPs on the boundary between dynamic/static region
    factory.reduce_slot_area(1, 1, dsp=100)
    factory.generate_virtual_device(Path(DEVICE_CONFIG))

    

    config = PipelineConfig(
        pp_scheme="double",
        pipeline_data_of_intra_slot_fifo="false",
        max_seconds="300",
        max_workers="1",
    )
    config.save_to_file(Path(PIPELINE_CONFIG))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate configuration files.")
    parser.add_argument(
        "--ch",
        type=int,
        required=True,
        help="The number of in/out channels.",
    )
    parser.add_argument(
        "--group_num",
        type=int,
        required=True,
        help="The number of NTT core groups.",
    )
    parser.add_argument(
        "--group_ch_num",
        type=int,
        required=True,
        help="The number of in/out channels per NTT core group.",
    )
    args = parser.parse_args()

    # Call the function with the provided ch
    gen_config(ch=args.ch, group_num=args.group_num, group_ch_num=args.group_ch_num)
