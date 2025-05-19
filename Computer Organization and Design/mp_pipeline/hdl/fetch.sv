module fetch
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,
    input   logic   [31:0]  imem_rdata,
    output  logic   [31:0]  imem_addr,
    output  logic   [3:0]   imem_rmask,
    output  if_id_stage_reg_t if_id, 
    input   logic           stall,
    input   logic   [31:0]  j_pc_next,
    output  logic           flush,
    input   logic           hold,
    input   logic           j
);

            logic           valid;
            logic   [31:0]  pc;
            logic   [31:0]  pc_next;
            logic   [31:0]  inst;

always_comb begin
        if_id.rvfi_data.valid = '0;
        if_id.rvfi_data.order = '0;
        if_id.rvfi_data.inst = '0;
        if_id.rvfi_data.rs1_addr = '0;
        if_id.rvfi_data.rs2_addr = '0;
        if_id.rvfi_data.rs1_rdata = '0;
        if_id.rvfi_data.rs2_rdata = '0;
        if_id.rvfi_data.regf_we = '0;
        if_id.rvfi_data.rd_addr = '0;
        if_id.rvfi_data.rd_wdata = '0;
        if_id.rvfi_data.pc_rdata = '0;
        if_id.rvfi_data.pc_wdata = '0;
        if_id.rvfi_data.mem_addr = '0;
        if_id.rvfi_data.mem_rmask = '0;
        if_id.rvfi_data.mem_wmask = '0;
        if_id.rvfi_data.mem_rdata = '0;
        if_id.rvfi_data.mem_wdata = '0;
        flush = '0;
        imem_rmask = '1;
        valid = '1;
        inst = imem_rdata;

        if(rst) begin
        pc_next = pc + 'd4;
        imem_addr = pc;
        end 
        else begin
            pc_next = pc + 'd4;
            imem_addr = pc_next;

            if(j) begin
                flush = '1;
                imem_addr = j_pc_next;
                pc_next = j_pc_next + 'd4;
            end
            if(hold || (~hold && stall)) begin
                imem_addr = pc;
            end

        if (~stall) begin
            if_id.rvfi_data.valid = valid;
            if_id.rvfi_data.pc_rdata = pc;
            if_id.rvfi_data.pc_wdata = pc_next;
            if_id.rvfi_data.inst = inst;
        end else begin
            if_id.rvfi_data.valid = '0;
            if_id.rvfi_data.pc_rdata = '0;
            if_id.rvfi_data.pc_wdata = '0;
            if_id.rvfi_data.inst = '0;
        end
    end
end

always_ff @ (posedge clk) begin
    if (rst) begin
        pc <= 32'haaaaa000;
    end else begin
        if (~stall && ~hold) begin
            pc <= ~j ? pc_next : j_pc_next;
        end
    end
end

endmodule : fetch