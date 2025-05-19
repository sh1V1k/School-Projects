module memory
import rv32i_types::*;
(
    input   ex_mm_stage_reg_t ex_mm,
    output  mm_wb_stage_reg_t mm_wb,
    input   logic   [31:0]  dmem_rdata,
    output  logic mem_inst,
    output   logic   [31:0] j_pc_next,
    output   logic          j,
    output   logic  [31:0]  forward
);

logic [31:0] temp;

always_comb begin
                mm_wb.rvfi_data.valid = '0;
                mm_wb.rvfi_data.order = '0;
                mm_wb.rvfi_data.inst = '0;
                mm_wb.rvfi_data.rs1_addr = '0;
                mm_wb.rvfi_data.rs2_addr = '0;
                mm_wb.rvfi_data.rs1_rdata = '0;
                mm_wb.rvfi_data.rs2_rdata = '0;
                mm_wb.rvfi_data.regf_we = '0;
                mm_wb.rvfi_data.rd_addr = '0;
                mm_wb.rvfi_data.rd_wdata = '0;
                mm_wb.rvfi_data.pc_rdata = '0;
                mm_wb.rvfi_data.pc_wdata = '0;
                mm_wb.rvfi_data.mem_addr = '0;
                mm_wb.rvfi_data.mem_rmask = '0;
                mm_wb.rvfi_data.mem_wmask = '0;
                mm_wb.rvfi_data.mem_rdata = '0;
                mm_wb.rvfi_data.mem_wdata = '0;
                mm_wb.bottom_two = '0;
                mem_inst = '0;
                j = '0;
                j_pc_next = '0;
                forward = '0;
                mm_wb.mem_inst = '0;

            if (ex_mm.rvfi_data.valid) begin
                j = ex_mm.j;
                j_pc_next = j ? ex_mm.rvfi_data.pc_wdata : '0;
                mm_wb.rvfi_data.valid = ex_mm.rvfi_data.valid;
                mm_wb.rvfi_data.order = ex_mm.rvfi_data.order;
                mm_wb.rvfi_data.pc_rdata = ex_mm.rvfi_data.pc_rdata;
                mm_wb.rvfi_data.pc_wdata = ex_mm.rvfi_data.pc_wdata;
                mm_wb.rvfi_data.inst = ex_mm.rvfi_data.inst;
                mm_wb.rvfi_data.rs1_addr = ex_mm.rvfi_data.rs1_addr;
                mm_wb.rvfi_data.rs2_addr = ex_mm.rvfi_data.rs2_addr;
                mm_wb.rvfi_data.rs1_rdata = ex_mm.rvfi_data.rs1_rdata;
                mm_wb.rvfi_data.rs2_rdata = ex_mm.rvfi_data.rs2_rdata;
                mm_wb.rvfi_data.regf_we = ex_mm.rvfi_data.regf_we;
                mm_wb.rvfi_data.rd_addr = ex_mm.rvfi_data.rd_addr;
                //mm_wb.rvfi_data.rd_wdata = ex_mm.rvfi_data.rd_wdata;
                mm_wb.rvfi_data.mem_addr = ex_mm.rvfi_data.mem_addr;
                mm_wb.rvfi_data.mem_rmask = ex_mm.rvfi_data.mem_rmask;
                mm_wb.rvfi_data.mem_wmask = ex_mm.rvfi_data.mem_wmask;
                //mm_wb.rvfi_data.mem_rdata = dmem_rdata; //<< 8*ex_mm.bottom_two
                mm_wb.rvfi_data.mem_wdata = ex_mm.rvfi_data.mem_wdata;
                mm_wb.bottom_two = ex_mm.bottom_two;

                mem_inst = ex_mm.mem_inst;

                unique case (ex_mm.rvfi_data.inst[14:12])
                        load_f3_lb : begin
                            temp = {{24{dmem_rdata[7 +8 *ex_mm.bottom_two[1:0]]}}, dmem_rdata[8 *ex_mm.bottom_two[1:0] +: 8 ]} << 8 *ex_mm.bottom_two[1:0];

                        end
                        load_f3_lbu: temp = {{24{1'b0}}                          , dmem_rdata[8 *ex_mm.bottom_two[1:0] +: 8 ]} << 8 *ex_mm.bottom_two[1:0];
                        load_f3_lh : temp = {{16{dmem_rdata[15+16*ex_mm.bottom_two[1]  ]}}, dmem_rdata[16*ex_mm.bottom_two[1]   +: 16]} << 8 *ex_mm.bottom_two[1:0];
                        load_f3_lhu: temp = {{16{1'b0}}                          , dmem_rdata[16*ex_mm.bottom_two[1]   +: 16]} << 8 *ex_mm.bottom_two[1:0];
                        load_f3_lw : temp = dmem_rdata;
                        default    : temp = 'x;
                endcase

                if (ex_mm.rvfi_data.mem_rmask > 0) begin
                    if(ex_mm.rvfi_data.inst[14:12] == load_f3_lhu || ex_mm.rvfi_data.inst[14:12] == load_f3_lbu) begin
                    mm_wb.rvfi_data.rd_wdata = temp >> 8*ex_mm.bottom_two; //>> 8*mm_wb.bottom_two
                    end else begin
                    mm_wb.rvfi_data.rd_wdata = unsigned'(signed'(temp) >>> 8*ex_mm.bottom_two);
                    end
                    //rd_v = mm_wb.rvfi_data.mem_rdata >> 8 *mm_wb.bottom_two[1:0];
                end else begin
                    mm_wb.rvfi_data.rd_wdata = ex_mm.rvfi_data.rd_wdata;
                end

                forward = mm_wb.rvfi_data.rd_wdata;

                mm_wb.rvfi_data.mem_rdata = mem_inst ? temp : '0;
                mm_wb.mem_inst = ex_mm.mem_inst;



            end
    end  
endmodule : memory