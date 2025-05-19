module execute
import rv32i_types::*;
(
    input   id_ex_stage_reg_t id_ex,
    output  ex_mm_stage_reg_t ex_mm,
    output logic [31:0] forward,
    input mm_wb_stage_reg_t wb,
    input ex_mm_stage_reg_t mm,
    output logic use_rd
);


    logic   [2:0]   funct3;
    logic   [6:0]   funct7;
    logic   [6:0]   opcode;
    logic   [31:0]  i_imm;
    logic   [31:0]  s_imm;
    logic   [31:0]  b_imm;
    logic   [31:0]  u_imm;
    logic   [31:0]  j_imm;
    logic   [31:0]  a;
    logic   [31:0]  b;


    logic           regf_we;
    logic   [31:0]  rd_v;

    logic   [2:0]   aluop;
    logic   [2:0]   cmpop;

    logic   [31:0]  aluout;
    logic           br_en;

    logic   [31:0] mem_addr;
    logic   [3:0]  mem_rmask;
    logic   [3:0]  mem_wmask;
    logic   [31:0] mem_wdata;

    logic   [1:0]  bottom_two;

    logic signed   [31:0] as;
    logic signed   [31:0] bs;
    logic unsigned [31:0] au;
    logic unsigned [31:0] bu;

    logic   [31:0]  rs1_v;
    logic   [31:0]  rs2_v;

    logic   [31:0] j_pc_next;
    logic          j;

    always_comb begin
        as =   '0;
      bs =   '0;
      au = '0;
      bu = '0;

          funct3 = '0;
      funct7 = '0;
      opcode = '0;
      i_imm = '0;
      s_imm = '0;
      b_imm = '0;
      u_imm = '0;
      j_imm = '0;
      regf_we =  '0;

      rs1_v = '0;
      rs2_v = '0;


        if(id_ex.rvfi_data.valid) begin
          as =   signed'(a);
      bs =   signed'(b);
      au = unsigned'(a);
      bu = unsigned'(b);
      rs1_v = id_ex.rvfi_data.rs1_rdata;
      rs2_v = id_ex.rvfi_data.rs2_rdata;

      if(wb.rvfi_data.rd_addr == id_ex.rvfi_data.rs1_addr && wb.rvfi_data.rd_addr != 0 && wb.rvfi_data.valid) begin
                    rs1_v = wb.rvfi_data.rd_wdata;
      end

      if(wb.rvfi_data.rd_addr == id_ex.rvfi_data.rs2_addr && wb.rvfi_data.rd_addr != 0 && wb.rvfi_data.valid) begin
                    rs2_v = wb.rvfi_data.rd_wdata;
      end

      if(mm.rvfi_data.rd_addr == id_ex.rvfi_data.rs1_addr && mm.rvfi_data.rd_addr != 0 && mm.rvfi_data.valid) begin
                    rs1_v = mm.rvfi_data.rd_wdata;
      end

      if(mm.rvfi_data.rd_addr == id_ex.rvfi_data.rs2_addr && mm.rvfi_data.rd_addr != 0 && mm.rvfi_data.valid) begin
                    rs2_v = mm.rvfi_data.rd_wdata;
      end 


          funct3 = id_ex.funct3;
      funct7 = id_ex.funct7;
      opcode = id_ex.opcode;
      i_imm = id_ex.i_imm;
      s_imm = id_ex.s_imm;
      b_imm = id_ex.b_imm;
      u_imm = id_ex.u_imm;
      j_imm = id_ex.j_imm;
      regf_we =  id_ex.rvfi_data.regf_we;
        end

    end

    always_comb begin
        aluout = '0;
        if(id_ex.rvfi_data.valid) begin
        unique case (aluop)
            alu_op_add: aluout = au +   bu;
            alu_op_sll: aluout = au <<  bu[4:0]; //its lower five bits for all shift instructions
            alu_op_sra: aluout = unsigned'(as >>> bu[4:0]);
            alu_op_sub: aluout = au -   bu;
            alu_op_xor: aluout = au ^   bu;
            alu_op_srl: aluout = au >>  bu[4:0];
            alu_op_or : aluout = au |   bu;
            alu_op_and: aluout = au &   bu;
            default   : aluout = 'x;
        endcase
        end
    end

    always_comb begin
        br_en = '0;
        if(id_ex.rvfi_data.valid) begin
        unique case (cmpop)
            branch_f3_beq : br_en = (au == bu);
            branch_f3_bne : br_en = (au != bu);
            branch_f3_blt : br_en = (as <  bs);
            branch_f3_bge : br_en = (as >=  bs); //changed to greater than equal vs before it was just greater than
            branch_f3_bltu: br_en = (au <  bu);
            branch_f3_bgeu: br_en = (au >=  bu);
            default       : br_en = 1'bx;
        endcase
        end
    end

    always_comb begin
        aluop = 'x;
        rd_v = 'x;
        cmpop = 'x;
        a = 'x;
        b = 'x;
        mem_addr = '0;
        mem_rmask = '0;
        mem_wmask = '0;
        mem_wdata = '0;
        use_rd = '1;
        bottom_two = '0;
        j_pc_next = '0;
        j = '0;
        if(id_ex.rvfi_data.valid) begin
        unique case(opcode)
        //op_b_jal       = 7'b1101111, // jump and link (J type)
        //op_b_jalr      = 7'b1100111, // jump and link register (I type)
        //op_b_br
            op_b_jal : begin
                rd_v = id_ex.rvfi_data.pc_wdata;
                j_pc_next = id_ex.rvfi_data.pc_rdata + j_imm;
                j = '1;
            end
            op_b_jalr : begin
                rd_v = id_ex.rvfi_data.pc_wdata;
                j = '1;
                j_pc_next = (rs1_v + i_imm) & 32'hfffffffe;
            end
            op_b_br : begin
                cmpop = funct3;
                a = rs1_v;
                b = rs2_v;
                use_rd = '0;
                if (br_en) begin
                j = '1;
                    j_pc_next = id_ex.rvfi_data.pc_rdata + b_imm;
                end
            end
            op_b_load : begin
                mem_addr = rs1_v + i_imm;
                unique case (funct3)
                    load_f3_lb, load_f3_lbu: mem_rmask = 4'b0001 << mem_addr[1:0];
                    load_f3_lh, load_f3_lhu: mem_rmask = 4'b0011 << mem_addr[1:0];
                    load_f3_lw             : mem_rmask = 4'b1111;
                    default                : mem_rmask = '0;
                endcase
            bottom_two = mem_addr[1:0];
            mem_addr[1:0] = 2'd0;
            end

            op_b_store : begin
                use_rd = '0;
                mem_addr = rs1_v + s_imm;
                unique case (funct3)
                    store_f3_sb: mem_wmask = 4'b0001 << mem_addr[1:0];
                    store_f3_sh: mem_wmask = 4'b0011 << mem_addr[1:0];
                    store_f3_sw: mem_wmask = 4'b1111;
                    default    : mem_wmask = 'x;
                endcase
                unique case (funct3)
                    store_f3_sb: mem_wdata[8 *mem_addr[1:0] +: 8 ] = rs2_v[7 :0];
                    store_f3_sh: mem_wdata[16*mem_addr[1]   +: 16] = rs2_v[15:0];
                    store_f3_sw: mem_wdata = rs2_v;
                    default    : mem_wdata = 'x;
                endcase
                bottom_two = mem_addr[1:0];
                mem_addr[1:0] = 2'd0;
            end

            op_b_auipc : begin
                rd_v = id_ex.rvfi_data.pc_rdata + u_imm;
            end
            op_b_lui : begin
                rd_v = u_imm;
            end
            op_b_imm  : begin
                a = rs1_v;
                b = i_imm;
                unique case (funct3)
                    arith_f3_or: begin
                        aluop = alu_op_or;
                        rd_v = aluout;
                    end
                    arith_f3_xor: begin
                        aluop = alu_op_xor;
                        rd_v = aluout;
                    end
                    arith_f3_sll: begin
                        aluop = alu_op_sll;
                        rd_v = aluout;
                    end
                    arith_f3_slt: begin
                        cmpop = branch_f3_blt;
                        rd_v = {31'd0, br_en};
                    end
                    arith_f3_sltu: begin
                        cmpop = branch_f3_bltu;
                        rd_v = {31'd0, br_en};
                    end
                    arith_f3_sr: begin
                        if (funct7[5]) begin
                            aluop = alu_op_sra;
                        end else begin
                            aluop = alu_op_srl;
                        end
                        rd_v = aluout;
                    end
                    arith_f3_add: begin
                        aluop = alu_op_add;
                        rd_v = aluout;
                    end
                    arith_f3_and: begin
                        aluop = alu_op_and;
                        rd_v = aluout;
                    end
                    default: begin
                        aluop = funct3;
                        rd_v = aluout;
                    end
                endcase
            end
            op_b_reg  : begin
                a = rs1_v;
                b = rs2_v;
                unique case (funct3)
                    arith_f3_or: begin
                        aluop = alu_op_or;
                        rd_v = aluout;
                    end
                    arith_f3_xor: begin
                        aluop = alu_op_xor;
                        rd_v = aluout;
                    end
                    arith_f3_sll: begin
                        aluop = alu_op_sll;
                        rd_v = aluout;
                    end
                    arith_f3_slt: begin
                        cmpop = branch_f3_blt;
                        rd_v = {31'd0, br_en};
                    end
                    arith_f3_sltu: begin
                        cmpop = branch_f3_bltu;
                        rd_v = {31'd0, br_en};
                    end
                    arith_f3_sr: begin
                        if (funct7[5]) begin
                            aluop = alu_op_sra;
                        end else begin
                            aluop = alu_op_srl;
                        end
                        rd_v = aluout;
                    end
                    arith_f3_add: begin
                        if (funct7[5]) begin
                            aluop = alu_op_sub;
                        end else begin
                            aluop = alu_op_add;
                        end
                        rd_v = aluout;
                    end
                    arith_f3_and: begin
                        aluop = alu_op_and;
                        rd_v = aluout;
                    end
                    default: begin
                        aluop = funct3;
                        rd_v = aluout;
                        cmpop = 'x;
                    end
                endcase
            end
            default   : begin
                aluop = funct3;
                rd_v = aluout;
            end
        endcase
        end
    end

    always_comb begin
                ex_mm.rvfi_data.valid = '0;
                ex_mm.rvfi_data.order = '0;
                ex_mm.rvfi_data.inst = '0;
                ex_mm.rvfi_data.rs1_addr = '0;
                ex_mm.rvfi_data.rs2_addr = '0;
                ex_mm.rvfi_data.rs1_rdata = '0;
                ex_mm.rvfi_data.rs2_rdata = '0;
                ex_mm.rvfi_data.regf_we = '0;
                ex_mm.rvfi_data.rd_addr = '0;
                ex_mm.rvfi_data.rd_wdata = '0;
                ex_mm.rvfi_data.pc_rdata = '0;
                ex_mm.rvfi_data.pc_wdata = '0;
                ex_mm.rvfi_data.mem_addr = '0;
                ex_mm.rvfi_data.mem_rmask = '0;
                ex_mm.rvfi_data.mem_wmask = '0;
                ex_mm.rvfi_data.mem_rdata = '0;
                ex_mm.rvfi_data.mem_wdata = '0;
                ex_mm.bottom_two = '0;
                ex_mm.j = '0;
                ex_mm.mem_inst = '0;
                forward = '0;

            if (id_ex.rvfi_data.valid) begin
                ex_mm.rvfi_data.rd_wdata = use_rd ? rd_v : '0;
                forward = use_rd ? rd_v : '0;

                if(mem_rmask > 0 || mem_wmask > 0) begin
                    forward = '0;
                    ex_mm.mem_inst = '1;
                end
                ex_mm.rvfi_data.rs1_rdata = rs1_v;
                ex_mm.rvfi_data.rs2_rdata = rs2_v;
                ex_mm.rvfi_data.regf_we =  (opcode == op_b_br) ? '0 : '1;

                ex_mm.rvfi_data.valid = id_ex.rvfi_data.valid;
                ex_mm.rvfi_data.order = id_ex.rvfi_data.order;
                ex_mm.rvfi_data.pc_rdata = id_ex.rvfi_data.pc_rdata;
                ex_mm.rvfi_data.pc_wdata = j ? j_pc_next : id_ex.rvfi_data.pc_wdata;
                ex_mm.rvfi_data.inst = id_ex.rvfi_data.inst;
                ex_mm.rvfi_data.rs1_addr = id_ex.rvfi_data.rs1_addr;
                ex_mm.rvfi_data.rs2_addr = id_ex.rvfi_data.rs2_addr;
                ex_mm.rvfi_data.rd_addr =  use_rd ? id_ex.rvfi_data.rd_addr : '0;

                ex_mm.rvfi_data.mem_addr = mem_addr;
                ex_mm.rvfi_data.mem_rmask = mem_rmask;
                ex_mm.rvfi_data.mem_wmask = mem_wmask;
                ex_mm.rvfi_data.mem_wdata = mem_wdata;
                ex_mm.bottom_two = bottom_two;
                ex_mm.j = j;
            end
    end  

endmodule : execute