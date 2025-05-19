/////////////////////////////////////////////////////////////
// Maybe merge what is in mp_verif/pkg/types.sv over here? //
/////////////////////////////////////////////////////////////

package rv32i_types;

    typedef enum logic {
        rs1_out = 1'b0,
        pc_out  = 1'b1
    } alu_m1_sel_t;

    // more mux def here

    typedef struct packed {
<<<<<<< HEAD
        logic           valid;
        logic   [63:0]  order;
        logic   [31:0]  inst;
        logic   [4:0]   rs1_addr;
        logic   [4:0]   rs2_addr;
        logic   [31:0]  rs1_rdata;
        logic   [31:0]  rs2_rdata;
        logic           regf_we;
        logic   [4:0]   rd_addr;
        logic   [31:0]  rd_wdata;
        logic   [31:0]  pc_rdata;
        logic   [31:0]  pc_wdata;
        logic   [31:0]  mem_addr;
        logic   [3:0]   mem_rmask;
        logic   [3:0]   mem_wmask;
        logic   [31:0]  mem_rdata;
        logic   [31:0]  mem_wdata;
    } rvfi_data_t;



    typedef struct packed {
        rvfi_data_t           rvfi_data;
    } if_id_stage_reg_t;

    typedef struct packed {
        //alu_m1_sel_t              alu_m1_sel;
        logic   [2:0]   funct3;
        logic   [6:0]   funct7;
        logic   [6:0]   opcode;
        logic   [31:0]  i_imm;
        logic   [31:0]  s_imm;
        logic   [31:0]  b_imm;
        logic   [31:0]  u_imm;
        logic   [31:0]  j_imm;
        rvfi_data_t               rvfi_data;
    } id_ex_stage_reg_t;

    typedef struct packed {
        rvfi_data_t           rvfi_data;
        logic [1:0]           bottom_two;
        logic                 j;
        logic mem_inst;
    } ex_mm_stage_reg_t;

    typedef struct packed {
        rvfi_data_t           rvfi_data;
        logic [1:0]           bottom_two;
        logic mem_inst;
    } mm_wb_stage_reg_t;



    typedef enum logic [6:0] {
        op_b_lui       = 7'b0110111, // load upper immediate (U type)
        op_b_auipc     = 7'b0010111, // add upper immediate PC (U type)
        op_b_jal       = 7'b1101111, // jump and link (J type)
        op_b_jalr      = 7'b1100111, // jump and link register (I type)
        op_b_br        = 7'b1100011, // branch (B type)
        op_b_load      = 7'b0000011, // load (I type)
        op_b_store     = 7'b0100011, // store (S type)
        op_b_imm       = 7'b0010011, // arith ops with register/immediate operands (I type)
        op_b_reg       = 7'b0110011  // arith ops with register operands (R type)
    } rv32i_opcode;

    typedef enum logic [2:0] {
        arith_f3_add   = 3'b000, // check logic 30 for sub if op_reg op
        arith_f3_sll   = 3'b001,
        arith_f3_slt   = 3'b010,
        arith_f3_sltu  = 3'b011,
        arith_f3_xor   = 3'b100,
        arith_f3_sr    = 3'b101, // check logic 30 for logical/arithmetic
        arith_f3_or    = 3'b110,
        arith_f3_and   = 3'b111
    } arith_f3_t;

    typedef enum logic [2:0] {
        load_f3_lb     = 3'b000,
        load_f3_lh     = 3'b001,
        load_f3_lw     = 3'b010,
        load_f3_lbu    = 3'b100,
        load_f3_lhu    = 3'b101
    } load_f3_t;

    typedef enum logic [2:0] {
        store_f3_sb    = 3'b000,
        store_f3_sh    = 3'b001,
        store_f3_sw    = 3'b010
    } store_f3_t;

    typedef enum logic [2:0] {
        branch_f3_beq  = 3'b000,
        branch_f3_bne  = 3'b001,
        branch_f3_blt  = 3'b100,
        branch_f3_bge  = 3'b101,
        branch_f3_bltu = 3'b110,
        branch_f3_bgeu = 3'b111
    } branch_f3_t;

    typedef enum logic [2:0] {
        alu_op_add     = 3'b000,
        alu_op_sll     = 3'b001,
        alu_op_sra     = 3'b010,
        alu_op_sub     = 3'b011,
        alu_op_xor     = 3'b100,
        alu_op_srl     = 3'b101,
        alu_op_or      = 3'b110,
        alu_op_and     = 3'b111
    } alu_ops;

    typedef enum logic [6:0] {
        base           = 7'b0000000,
        variant        = 7'b0100000
    } funct7_t;

    typedef union packed {
        logic [31:0] word;

        struct packed {
            logic [11:0] i_imm;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  rd;
            rv32i_opcode opcode;
        } i_type;

        struct packed {
            logic [6:0]  funct7;
            logic [4:0]  rs2;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  rd;
            rv32i_opcode opcode;
        } r_type;

        struct packed {
            logic [11:5] imm_s_top;
            logic [4:0]  rs2;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  imm_s_bot;
            rv32i_opcode opcode;
        } s_type;


        struct packed {
            logic imm_12;
            logic [10:5] imm_top;
            logic [4:0] rs2;
            logic [4:0] rs1;
            logic [2:0] funct3;
            logic [4:1] imm_bot;
            logic imm_11;
            rv32i_opcode opcode;
        } b_type;

        struct packed {
            logic [31:12] imm;
            logic [4:0]   rd;
            rv32i_opcode  opcode;
        } j_type;

    } instr_t;

=======
        logic   [31:0]      inst;
        logic   [31:0]      pc;
        logic   [63:0]      order;

        alu_m1_sel_t        alu_m1_sel;

        // what else?

    } id_ex_stage_reg_t;

>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
endpackage
