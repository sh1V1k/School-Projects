module writeback
import rv32i_types::*;
(
    input   mm_wb_stage_reg_t mm_wb,
    input   logic           rst,
    input   logic           clk,
    output  logic           monitor_valid,
    output  logic   [63:0]  monitor_order,
    output  logic   [31:0]  monitor_inst,
    output  logic   [4:0]   monitor_rs1_addr,
    output  logic   [4:0]   monitor_rs2_addr,
    output  logic   [31:0]  monitor_rs1_rdata,
    output  logic   [31:0]  monitor_rs2_rdata,
    output  logic           monitor_regf_we,
    output  logic   [4:0]   monitor_rd_addr,
    output  logic   [31:0]  monitor_rd_wdata,
    output  logic   [31:0]  monitor_pc_rdata,
    output  logic   [31:0]  monitor_pc_wdata,
    output  logic   [31:0]  monitor_mem_addr,
    output  logic   [3:0]   monitor_mem_rmask,
    output  logic   [3:0]   monitor_mem_wmask,
    output  logic   [31:0]  monitor_mem_rdata,
    output  logic   [31:0]  monitor_mem_wdata,

    output  logic   [4:0]   rd_s,
    output  logic           regf_we,
    output  logic   [31:0]  rd_v,

    input   logic stall
);
    logic [63:0] order;

    always_ff @ (posedge clk) begin
        if (rst) begin
            order <= '0;
        end else begin
            if(mm_wb.rvfi_data.valid && ~stall) begin
                order <= order + 'd1;
            end
        end
    end

    always_comb begin
        rd_s  = '0;
         regf_we = '0;
         rd_v = '0;

         monitor_valid     = '0;
         monitor_order     = '0;
         monitor_inst      = '0;
         monitor_rs1_addr  = '0;
         monitor_rs2_addr  = '0;
         monitor_rs1_rdata = '0;
         monitor_rs2_rdata = '0;
         monitor_regf_we   = '0;
         monitor_rd_addr   = '0;
         monitor_rd_wdata  = '0;
         monitor_pc_rdata  = '0;
         monitor_pc_wdata  = '0;
         monitor_mem_addr  = '0;
         monitor_mem_rmask = '0;
         monitor_mem_wmask = '0;
         monitor_mem_rdata = '0;
         monitor_mem_wdata = '0;

        if (~stall) begin
         rd_s  = mm_wb.rvfi_data.regf_we ? mm_wb.rvfi_data.rd_addr : 5'd0;

         regf_we = mm_wb.rvfi_data.regf_we;

         rd_v = mm_wb.rvfi_data.rd_wdata;



        
         monitor_valid     = mm_wb.rvfi_data.valid;
         monitor_order     = order;
         monitor_inst      = mm_wb.rvfi_data.inst;
         monitor_rs1_addr  = mm_wb.rvfi_data.rs1_addr;
         monitor_rs2_addr  = mm_wb.rvfi_data.rs2_addr;
         monitor_rs1_rdata = mm_wb.rvfi_data.rs1_rdata;
         monitor_rs2_rdata = mm_wb.rvfi_data.rs2_rdata;
         monitor_regf_we   = mm_wb.rvfi_data.regf_we;
         monitor_rd_addr   = mm_wb.rvfi_data.regf_we ? mm_wb.rvfi_data.rd_addr : 5'd0;
         monitor_rd_wdata  = mm_wb.rvfi_data.rd_wdata;
         monitor_pc_rdata  = mm_wb.rvfi_data.pc_rdata;
         monitor_pc_wdata  = mm_wb.rvfi_data.pc_wdata;
         monitor_mem_addr  = mm_wb.rvfi_data.mem_addr;
         monitor_mem_rmask = mm_wb.rvfi_data.mem_rmask;
         monitor_mem_wmask = mm_wb.rvfi_data.mem_wmask;
         monitor_mem_rdata = mm_wb.rvfi_data.mem_rdata;
         monitor_mem_wdata = mm_wb.rvfi_data.mem_wdata;

        end

    end

endmodule : writeback
