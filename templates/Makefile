# Define variables
APP := ntt
HOST := host_ntt

# Default TARGET is 'sw', and it can be overridden by specifying a different value
TARGET := sw

KERNEL_FILE := ./src/ntt.cpp
HOST_FILE := ./src/host.cpp
CXX_FLAGS := -O2 -fopenmp

PART_NUM := xcu55c-fsvh2892-2L-e
CLOCK_PERIOD := 3.33
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1

SIM_XO_DIR := ./sim_xo
SIM_XO_OPT_DIR := ./sim_xo_opt
OPT_DIR := ./opt
CONFIG_BUILD_DIR := $(OPT_DIR)/config_build
OPT_WORK_DIR := ./x_opt
HW_WORK_DIR := ./x_hw

XO_FILE := $(APP).xo
BITSTREAM_HW := $(APP).$(PLATFORM).hw.xclbin
BITSTREAM_HW_OPT := $(APP)_$(PLATFORM).xclbin

HW_OUTPUT := ./hw
HW_OPT_OUTPUT := ./hw_opt

# Default is to save waveform
SAVE_WAVEFORM := 0 

MAKEFLAGS += --no-print-directory

CH := {CH} 
GROUP_NUM := {GROUP_NUM}
GROUP_CH_NUM := {GROUP_CH_NUM}

# Check if NUM was set by the user or not

#NUM := 100 
ifeq ($(origin NUM), undefined)
    SIM_NUM := $(shell expr 50 \* $(CH))
    HW_SIM_NUM := $(shell expr 10000 \* $(CH))
else
    SIM_NUM := $(NUM)
    HW_SIM_NUM := $(NUM)
endif

.PHONY: all compile run clean help

VALID_TARGETS := sw xo xo-opt hw hw-opt

# Build target
all:
	@if echo "$(VALID_TARGETS)" | grep -wq "$(TARGET)"; then \
		$(MAKE) build-$(TARGET); \
	else \
		echo "Invalid TARGET: $(TARGET). Use one of: $(VALID_TARGETS)."; \
		exit 1; \
	fi

# Run target
run:
	@if echo "$(VALID_TARGETS)" | grep -wq "$(TARGET)"; then \
		$(MAKE) run-$(TARGET); \
	else \
		echo "Invalid TARGET: $(TARGET). Use one of: $(VALID_TARGETS)."; \
		exit 1; \
	fi

# Compile the application
build-sw compile:
	@echo "Compiling the application..."
	tapa g++ $(KERNEL_FILE) $(HOST_FILE) -o $(HOST) $(CXX_FLAGS)
	@if [ $$? -eq 0 ]; then \
		echo "Compile successful!"; \
	else \
		echo "Compile failed!" >&2; \
		exit 1; \
	fi

# Run the compiled application @echo "Running the compiled application..."
run-sw:
	@if [ ! -f $(HOST) ]; then \
		echo "Error: Executable $(HOST) not found. Please build it first!" >&2; \
		exit 1; \
	fi
	./$(HOST) $(SIM_NUM)

# Build XO file
build-xo:
	@echo "Building the XO file..."
		tapa compile \
			--top $(APP) \
			--part-num $(PART_NUM) \
			--clock-period $(CLOCK_PERIOD) \
			-f $(KERNEL_FILE) \
			-o $(XO_FILE)
	@echo "XO file generation complete."

# Run XO simulation
run-xo:
	@echo "Running the XO file simulation..."
	@if [ "$(SAVE_WAVEFORM)" -eq 1 ]; then \
		./$(HOST) --bitstream $(XO_FILE) \
			--xosim-part-num $(PART_NUM) \
			--xosim_work_dir $(SIM_XO_DIR) \
			--xosim_save_waveform \
			$(SIM_NUM); \
	else \
		./$(HOST) --bitstream $(XO_FILE) \
			--xosim-part-num $(PART_NUM) \
			--xosim_work_dir $(SIM_XO_DIR) \
			$(SIM_NUM); \
	fi
	@echo "XO simulation complete."

# Run XO GUI simulation
run-xo-gui:
	vivado -mode gui -source $(SIM_XO_DIR)/output/run/run_cosim.tcl

# Optimize design and build xo file
build-xo-opt:
	@echo "Making configuration files for rapidstream..."
	mkdir -p $(OPT_DIR)
	(cd $(OPT_DIR) && rapidstream ../gen_config.py --ch $(CH) --group_num $(GROUP_NUM) --group_ch_num $(GROUP_CH_NUM))
	@echo "Optimizing the design..."
	rapidstream-tapaopt \
		--work-dir $(OPT_DIR) \
		--tapa-xo-path $(XO_FILE) \
		--device-config $(CONFIG_BUILD_DIR)/device_config.json \
		--pipeline-config $(CONFIG_BUILD_DIR)/pipeline_config.json \
		--floorplan-config $(CONFIG_BUILD_DIR)/floorplan_config.json \
		--connectivity-ini ./link_config_1.ini
	@echo "Optimization complete."

# Optimize design and build FPGA bitstream
build-hw-opt:
	@echo "Making configuration files for rapidstream..."
	mkdir -p $(OPT_DIR)
	(cd $(OPT_DIR) && rapidstream ../gen_config.py --ch $(CH) --group_num $(GROUP_NUM) --group_ch_num $(GROUP_CH_NUM))
	@echo "Optimizing the design..."
	rapidstream-tapaopt \
		--work-dir $(OPT_DIR) \
		--tapa-xo-path $(XO_FILE) \
		--device-config $(CONFIG_BUILD_DIR)/device_config.json \
		--pipeline-config $(CONFIG_BUILD_DIR)/pipeline_config.json \
		--floorplan-config $(CONFIG_BUILD_DIR)/floorplan_config.json \
		--implementation-config ./impl_config.json \
		--connectivity-ini ./link_config.ini \
		--run-impl || true
	@echo "Searching generated bitstreams"
	@for i in 0 1 2 3 4; do \
		if [ -f "$(OPT_DIR)/dse/solution_$${i}/vitis_run_hw/$(BITSTREAM_HW_OPT)" ]; then \
			cp "$(OPT_DIR)/dse/solution_$${i}/vitis_run_hw/$(BITSTREAM_HW_OPT)" "$(OPT_DIR)/$(BITSTREAM_HW_OPT)"; \
			echo "Bitstream found: solution_$${i}. Copying to $(OPT_DIR)/$(BITSTREAM_HW_OPT)."; \
			break; \
		fi; \
	done

# Run optimized design
run-xo-opt:
	@echo "Running the optimized design..."
	@if [ "$(SAVE_WAVEFORM)" -eq 1 ]; then \
		./$(HOST) \
			--bitstream "$(OPT_DIR)/dse/solution_0/updated.xo" \
			--xosim-part-num $(PART_NUM) \
			--xosim_work_dir $(SIM_XO_OPT_DIR) \
			--xosim_save_waveform \
			$(SIM_NUM); \
	else \
		./$(HOST) \
			--bitstream "$(OPT_DIR)/dse/solution_0/updated.xo" \
			--xosim-part-num $(PART_NUM) \
			--xosim_work_dir $(SIM_XO_OPT_DIR) \
			$(SIM_NUM); \
	fi
	@echo "Optimized design run complete."

# Build the FPGA bitstream
build-hw:
	@echo "Building the hardware configuration..."
	v++ \
		-o $(HW_OUTPUT)/$(BITSTREAM_HW) \
		--temp_dir $(HW_WORK_DIR) \
		--link \
		--target hw \
		--kernel $(APP) \
		--platform $(PLATFORM) \
		--config link_config_1.ini \
		$(XO_FILE)
	@echo "Hardware configuration build complete."

build-hw-opt-old:
	@echo "Building the hardware configuration..."
	v++ \
		-o $(HW_OPT_OUTPUT)/$(BITSTREAM_HW) \
		--temp_dir $(OPT_WORK_DIR) \
		--link \
		--target hw \
		--kernel $(APP) \
		--platform $(PLATFORM) \
		--config link_config.ini \
		$(OPT_DIR)/dse/solution_0/updated.xo
	@echo "Hardware configuration build complete."

# Run on FPGA
run-hw:
	@echo "Running on hardware..."
	./$(HOST) --bitstream $(HW_OUTPUT)/$(BITSTREAM_HW) $(HW_SIM_NUM)
	@echo "Hardware run complete."

run-hw-opt:
	@echo "Running on hardware..."
	./$(HOST) --bitstream $(OPT_DIR)/$(BITSTREAM_HW_OPT) $(HW_SIM_NUM)
	@echo "Hardware run complete."

run-hw-opt-old:
	@echo "Running on hardware..."
	./$(HOST) --bitstream $(HW_OPT_OUTPUT)/$(BITSTREAM_HW) $(HW_SIM_NUM)
	@echo "Hardware run complete."

#Help
help:
	@echo "Usage:"
	@echo "  Build Commands:"
	@echo "    make all TARGET=<type>  # Build the specified TARGET"
	@echo "      Valid TARGET values:"
	@echo "        sw      - Build the host executable"
	@echo "        xo      - Build the kernel XO file"
	@echo "        hw      - Build the FPGA bitstream"
	@echo "        hw-opt  - Optimize with Rapidstream and build FPGA bitstream"
	@echo ""
	@echo "  Run Commands:"
	@echo "    make run TARGET=<type> NUM=<poly_num>  # Run with the TARGET for specified number of polynomials"
	@echo "      Valid TARGET values:"
	@echo "        sw      - Run software simulation"
	@echo "        xo      - Simulate with the XO file"
	@echo "        hw      - Run the FPGA bitstream"
	@echo "        hw-opt  - Run the RS-optimized FPGA bitstream"
	@echo ""

# Clean generated files
clean:
	@echo "Cleaning up..."
	@rm -f $(HOST) $(XO_FILE)
	@rm -rf work.out .ipcache .Xil $(OPT_DIR) $(HW_WORK_DIR) $(HW_OUTPUT) $(OPT_WORK_DIR) $(HW_OPT_OUTPUT)
	@rm -rf $(SIM_XO_DIR) $(SIM_XO_OPT_DIR)
	@rm -f v++*.log xcd.log gurobi.log vivado.log vivado.jou
	@echo "Clean up complete."
