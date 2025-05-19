module decode
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,
    input   if_id_stage_reg_t if_id,
    output  id_ex_stage_reg_t id_ex,
    input   logic [4:0] rd_s,
    input   logic [31:0] rd_v,
    input logic regf_we,
    input id_ex_stage_reg_t ex,
    input ex_mm_stage_reg_t mm,
    input logic hold,
    input logic  [31:0] ex_forward,
    input logic  [31:0] mem_forward,
    input logic use_rd
);

    logic   [31:0]  local_inst;
    logic   [2:0]   funct3;
    logic   [6:0]   funct7;
    logic   [6:0]   opcode;
    logic   [31:0]  i_imm;
    logic   [31:0]  s_imm;
    logic   [31:0]  b_imm;
    logic   [31:0]  u_imm;
    logic   [31:0]  j_imm;
    logic   [4:0]   rs1_s;
    logic   [4:0]   decode_rd_s;
    logic   [4:0]   rs2_s;
    logic   [31:0]  rs1_v;
    logic   [31:0]  rs2_v;

    logic   [31:0]  rs1_v_forward;
    logic   [31:0]  rs2_v_forward;


    assign local_inst = if_id.rvfi_data.inst;
    assign funct3 = local_inst[14:12];
    assign funct7 = local_inst[31:25];
    assign opcode = local_inst[6:0];
    assign i_imm  = {{21{local_inst[31]}}, local_inst[30:20]};
    assign s_imm  = {{21{local_inst[31]}}, local_inst[30:25], local_inst[11:7]};
    assign b_imm  = {{20{local_inst[31]}}, local_inst[7], local_inst[30:25], local_inst[11:8], 1'b0};
    assign u_imm  = {local_inst[31:12], 12'h000};
    assign j_imm  = {{12{local_inst[31]}}, local_inst[19:12], local_inst[20], local_inst[30:21], 1'b0};
    assign rs1_s  = local_inst[19:15];
    assign rs2_s  = local_inst[24:20];
    assign decode_rd_s   = local_inst[11:7];

    regfile regfile(
                .clk(clk),
                .rst(rst),
                .regf_we(regf_we),
                .rs1_s(rs1_s),
                .rs2_s(rs2_s),
                .rd_s(rd_s),
                .rs1_v(rs1_v),
                .rs2_v(rs2_v),
                .rd_v(rd_v)
            );

    always_comb begin
        rs1_v_forward = rs1_v;
        rs2_v_forward = rs2_v;
        
        //mem to decode
        if(mm.rvfi_data.rd_addr == rs1_s && mm.rvfi_data.rd_addr != 0 && mm.rvfi_data.valid && ~hold) begin
                    rs1_v_forward = ~hold ? mem_forward : '1;
        end
        if(mm.rvfi_data.rd_addr == rs2_s && mm.rvfi_data.rd_addr != 0 && mm.rvfi_data.valid && ~hold) begin
                    rs2_v_forward = ~hold ? mem_forward : '1;
        end

        //ex to decode forwarding
        if(use_rd && ex.rvfi_data.rd_addr == rs1_s && ex.rvfi_data.rd_addr != 0 && ex.rvfi_data.valid && ~hold) begin
                    rs1_v_forward = ~hold ? ex_forward : '1;
        end
        if(use_rd && ex.rvfi_data.rd_addr == rs2_s && ex.rvfi_data.rd_addr != 0 && ex.rvfi_data.valid && ~hold) begin
                    rs2_v_forward = ~hold ? ex_forward : '1;
        end

    end

    always_comb begin
        id_ex.rvfi_data.rs1_addr = '0;
        id_ex.rvfi_data.rs2_addr = '0;
        id_ex.rvfi_data.rs1_rdata = '0;
        id_ex.rvfi_data.rs2_rdata = '0;
        id_ex.rvfi_data.regf_we = '0;
        id_ex.rvfi_data.rd_addr = '0;
        id_ex.rvfi_data.rd_wdata = '0;
        id_ex.funct3 = '0;
        id_ex.funct7 = '0;
        id_ex.opcode = '0;
        id_ex.i_imm = '0;
        id_ex.s_imm = '0;
        id_ex.b_imm = '0;
        id_ex.u_imm = '0;
        id_ex.j_imm = '0;

        id_ex.rvfi_data.valid = '0;
        id_ex.rvfi_data.order = '0;
        id_ex.rvfi_data.pc_rdata = '0;
        id_ex.rvfi_data.pc_wdata = '0;
        id_ex.rvfi_data.inst = '0;


        id_ex.rvfi_data.mem_addr = '0;
        id_ex.rvfi_data.mem_rmask = '0;
        id_ex.rvfi_data.mem_wmask = '0;
        id_ex.rvfi_data.mem_rdata = '0;
        id_ex.rvfi_data.mem_wdata = '0;

            if (if_id.rvfi_data.valid) begin
                id_ex.rvfi_data.rs1_addr = rs1_s;
                id_ex.rvfi_data.rs2_addr = rs2_s;
                id_ex.rvfi_data.rs1_rdata = rs1_v_forward;
                id_ex.rvfi_data.rs2_rdata = rs2_v_forward;
                id_ex.rvfi_data.rd_addr = decode_rd_s;
                id_ex.funct3 = funct3;
                id_ex.funct7 = funct7;
                id_ex.opcode = opcode;
                id_ex.i_imm = i_imm;
                id_ex.s_imm = s_imm;
                id_ex.b_imm = b_imm;
                id_ex.u_imm = u_imm;
                id_ex.j_imm = j_imm;

                id_ex.rvfi_data.valid = if_id.rvfi_data.valid;
                id_ex.rvfi_data.order = if_id.rvfi_data.order;
                id_ex.rvfi_data.pc_rdata = if_id.rvfi_data.pc_rdata;
                id_ex.rvfi_data.pc_wdata = if_id.rvfi_data.pc_wdata;
                id_ex.rvfi_data.inst = if_id.rvfi_data.inst;
            end
    end            
    

endmodule : decode