module cpu
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,

    output  logic   [31:0]  imem_addr,
    output  logic   [3:0]   imem_rmask,
    input   logic   [31:0]  imem_rdata,
    input   logic           imem_resp,

    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    input   logic   [31:0]  dmem_rdata,
    output  logic   [31:0]  dmem_wdata,
    input   logic           dmem_resp
);

<<<<<<< HEAD
            logic           monitor_valid;
            logic   [63:0]  monitor_order;
            logic   [31:0]  monitor_inst;
            logic   [4:0]   monitor_rs1_addr;
            logic   [4:0]   monitor_rs2_addr;
            logic   [31:0]  monitor_rs1_rdata;
            logic   [31:0]  monitor_rs2_rdata;
            logic           monitor_regf_we;
            logic   [4:0]   monitor_rd_addr;
            logic   [31:0]  monitor_rd_wdata;
            logic   [31:0]  monitor_pc_rdata;
            logic   [31:0]  monitor_pc_wdata;
            logic   [31:0]  monitor_mem_addr;
            logic   [3:0]   monitor_mem_rmask;
            logic   [3:0]   monitor_mem_wmask;
            logic   [31:0]  monitor_mem_rdata;
            logic   [31:0]  monitor_mem_wdata;

            logic   [4:0]   rd_s;
            logic use_rd;

            logic           regf_we;
            logic   [31:0]  rd_v;
            logic   [63:0]  order;

            logic stall;
            logic mem_inst;
            logic j;

            logic [31:0] j_pc_next;
            logic        flush;
            logic           hold;

            logic  [31:0] ex_forward;
            logic  [31:0] mm_forward;



    id_ex_stage_reg_t id_ex_stage_reg, id_ex_stage_reg_next;

    if_id_stage_reg_t if_id_stage_reg, if_id_stage_reg_next;

    ex_mm_stage_reg_t ex_mm_stage_reg, ex_mm_stage_reg_next;

    mm_wb_stage_reg_t mm_wb_stage_reg, mm_wb_stage_reg_next;


    always_comb begin
        dmem_addr = '0;
        dmem_rmask = '0;
        dmem_wmask = '0;
        dmem_wdata = '0;
        if(~imem_resp || hold || flush) begin
            if(ex_mm_stage_reg.rvfi_data.valid) begin
            dmem_addr = ex_mm_stage_reg.rvfi_data.mem_addr;
            dmem_rmask = ex_mm_stage_reg.rvfi_data.mem_rmask;
            dmem_wmask = ex_mm_stage_reg.rvfi_data.mem_wmask;
            dmem_wdata = ex_mm_stage_reg.rvfi_data.mem_wdata;
            end
        end else begin
            if(ex_mm_stage_reg_next.rvfi_data.valid) begin
            dmem_addr = ex_mm_stage_reg_next.rvfi_data.mem_addr;
            dmem_rmask = ex_mm_stage_reg_next.rvfi_data.mem_rmask;
            dmem_wmask = ex_mm_stage_reg_next.rvfi_data.mem_wmask;
            dmem_wdata = ex_mm_stage_reg_next.rvfi_data.mem_wdata;
            end
        end
    end

    assign stall = (~imem_resp) || ((~dmem_resp) && mem_inst);

    always_comb begin
    hold = '0;
    if(ex_mm_stage_reg.rvfi_data.rd_addr != 0 && ex_mm_stage_reg.rvfi_data.valid && ex_mm_stage_reg.rvfi_data.mem_rmask > 0 &&((ex_mm_stage_reg.rvfi_data.rd_addr == id_ex_stage_reg.rvfi_data.rs1_addr) || 
                (ex_mm_stage_reg.rvfi_data.rd_addr == id_ex_stage_reg.rvfi_data.rs2_addr))) begin
                    hold = '1;
                end
    end

    always_ff @(posedge clk) begin
        if(rst)begin
        if_id_stage_reg <= '0;
        id_ex_stage_reg <= '0;
        ex_mm_stage_reg <= '0;
        mm_wb_stage_reg <= '0;
        end 
        else begin
            if(~stall) begin
                if(hold) begin
                    ex_mm_stage_reg.rvfi_data.valid <= '0;
                    mm_wb_stage_reg <= mm_wb_stage_reg_next;
                end else begin
                if_id_stage_reg <= if_id_stage_reg_next;
                id_ex_stage_reg <= id_ex_stage_reg_next;
                ex_mm_stage_reg <= ex_mm_stage_reg_next;
                mm_wb_stage_reg <= mm_wb_stage_reg_next;
                end
                if(flush && ~hold) begin
                    if_id_stage_reg.rvfi_data.valid <= '0;
                    id_ex_stage_reg.rvfi_data.valid <= '0;
                    ex_mm_stage_reg.rvfi_data.valid <= '0;
                end
            end
        end
    end

fetch f_stage(.clk(clk), .rst(rst), .if_id(if_id_stage_reg_next), .imem_rdata(imem_rdata), .imem_addr(imem_addr), .imem_rmask(imem_rmask), .stall(stall), .j_pc_next(j_pc_next), .flush(flush), .hold(hold), .j(j));

decode d_stage(.if_id(if_id_stage_reg), .id_ex(id_ex_stage_reg_next), .rd_s(rd_s), .regf_we(regf_we), .rd_v(rd_v), .clk(clk), .rst(rst), .ex(id_ex_stage_reg), .hold(hold), .mm(ex_mm_stage_reg), .ex_forward(ex_forward), .use_rd(use_rd), .mem_forward(mm_forward));

execute e_stage(.id_ex(id_ex_stage_reg), .ex_mm(ex_mm_stage_reg_next), .forward(ex_forward), .wb(mm_wb_stage_reg), .mm(ex_mm_stage_reg), .use_rd(use_rd)); //.dmem_addr(dmem_addr), .dmem_rmask(dmem_rmask), .dmem_wmask(dmem_wmask), .dmem_wdata(dmem_wdata)

memory m_stage(.ex_mm(ex_mm_stage_reg), .mm_wb(mm_wb_stage_reg_next), .dmem_rdata(dmem_rdata), .mem_inst(mem_inst), .j_pc_next(j_pc_next), .j(j), .forward(mm_forward));

writeback wb_stage(.clk(clk), .rst(rst), .mm_wb(mm_wb_stage_reg), .monitor_valid(monitor_valid), .monitor_order(monitor_order), .monitor_inst(monitor_inst), .monitor_rs1_addr(monitor_rs1_addr), .monitor_rs2_addr(monitor_rs2_addr), .monitor_rs1_rdata(monitor_rs1_rdata),
.monitor_rs2_rdata(monitor_rs2_rdata), .monitor_rd_addr(monitor_rd_addr), .monitor_rd_wdata(monitor_rd_wdata), .monitor_pc_rdata(monitor_pc_rdata), .monitor_pc_wdata(monitor_pc_wdata),
.monitor_mem_addr(monitor_mem_addr), .monitor_mem_rmask(monitor_mem_rmask), .monitor_mem_wmask(monitor_mem_wmask), .monitor_mem_rdata(monitor_mem_rdata), .monitor_mem_wdata(monitor_mem_wdata),
.rd_s(rd_s), .regf_we(regf_we), .rd_v(rd_v), .stall(stall));
// .dmem_addr(dmem_addr), .dmem_rmask(dmem_rmask), .dmem_wmask(dmem_wmask), .dmem_wdata(dmem_wdata), .stall(stall));


=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
endmodule : cpu
