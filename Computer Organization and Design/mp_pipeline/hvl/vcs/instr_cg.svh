covergroup instr_cg with function sample(instr_t instr);
    // Easy covergroup to see that we're at least exercising
    // every opcode. Since opcode is an enum, this makes bins
    // for all its members.
    all_opcodes : coverpoint instr.i_type.opcode;


    // Some simple coverpoints on various instruction fields.
    // Recognize that these coverpoints are inherently less useful
    // because they really make sense in the context of the opcode itself.
    all_funct7 : coverpoint funct7_t'(instr.r_type.funct7);

    // TODO: Write the following coverpoints:

    // Check that funct3 takes on all possible values.
    // all_funct3 : coverpoint ... ;
    all_funct3 : coverpoint instr.r_type.funct3; //don't know if this is right, confirm with CA

    // Check that the rs1 and rs2 fields across instructions take on
    // all possible values (each register is touched).
    // all_regs_rs1 : coverpoint ... ;
    // all_regs_rs2 : coverpoint ... ;
    // Coverpoint for rs1
// Coverpoint for rs1
    all_regs_rs1 : coverpoint instr.r_type.rs1; //don't know if this is right, confirm with CA
    all_regs_rs2 : coverpoint instr.r_type.rs2; //don't know if this is right, confirm with CA


    // Now, cross coverage takes in the opcode context to correctly
    // figure out the /actual/ coverage.
    funct3_cross : cross instr.i_type.opcode, instr.i_type.funct3 {

        // We want to ignore the cases where funct3 isn't relevant.

        // For example, for JAL, funct3 doesn't exist. Put it in an ignore_bins.
        ignore_bins JAL_FUNCT3 = funct3_cross with (instr.i_type.opcode == op_b_jal);

        // TODO:    What other opcodes does funct3 not exist for? Put those in
        // ignore_bins.
        ignore_bins AUIPC_FUNCT3 = funct3_cross with (instr.i_type.opcode == op_b_auipc);
        ignore_bins LUI_FUNCT3 = funct3_cross with (instr.i_type.opcode == op_b_lui);


        // Branch instructions use funct3, but only 6 of the 8 possible values
        // are valid. Ignore the other two -- don't add them into the coverage
        // report. In fact, if they're generated, that's an illegal instruction.
        illegal_bins BR_FUNCT3 = funct3_cross with
        (instr.i_type.opcode == op_b_br
        && !(instr.i_type.funct3 inside {branch_f3_beq, branch_f3_bne, branch_f3_blt, branch_f3_bge, branch_f3_bltu, branch_f3_bgeu}));

        // TODO: You'll also have to ignore some funct3 cases in JALR, LOAD, and
        // STORE. Write the illegal_bins/ignore_bins for those cases.

        illegal_bins JALR_FUNCT3 = funct3_cross with (instr.i_type.opcode == op_b_jalr && !(instr.i_type.funct3 inside {3'b000}));

        illegal_bins STR_FUNCT3 = funct3_cross with 
        (instr.i_type.opcode == op_b_store
        && !(instr.i_type.funct3 inside {store_f3_sb, store_f3_sh, store_f3_sw}));

        illegal_bins LD_FUNCT3 = funct3_cross with 
        (instr.i_type.opcode == op_b_load
        && !(instr.i_type.funct3 inside {load_f3_lb, load_f3_lh, load_f3_lw, load_f3_lbu, load_f3_lhu}));
    }

    // Coverpoint to make separate bins for funct7.
    coverpoint instr.r_type.funct7 {
        bins range[] = {[0:$]};
        ignore_bins not_in_spec = {[1:31], [33:127]};
    }

    // Cross coverage for funct7. DONE!!!!!
    funct7_cross : cross instr.r_type.opcode, instr.r_type.funct3, instr.r_type.funct7 {

        // No opcodes except op_b_reg and op_b_imm use funct7, so ignore the rest.
        ignore_bins OTHER_INSTS = funct7_cross with
        (!(instr.r_type.opcode inside {op_b_reg, op_b_imm}));

        // TODO: Get rid of all the other cases where funct7 isn't necessary, or cannot
        // take on certain values.
        // ignore_bins ARITH_F7 = funct7_cross with
        // (instr.r_type.opcode == op_b_reg && !(instr.r_type.funct3 inside {3'b000, 3'b001, 3'b010, 3'b011, 3'b100, 3'b101, 3'b110, 3'b111}));

        ignore_bins SHIFT_F7 = funct7_cross with 
        (instr.r_type.opcode == op_b_imm && !(instr.r_type.funct3 inside {3'b001, 3'b101}));

        //covers all reg instructions that use funct7
        illegal_bins NO_VARIANT_IMM_F7 = funct7_cross with 
        (instr.r_type.funct7 == 32 && instr.r_type.opcode == op_b_reg
        && !(instr.r_type.funct3 inside {3'b101, 3'b000}));

        illegal_bins IMM_CASE2 = funct7_cross with 
        (instr.r_type.funct7 == 32 && instr.r_type.opcode == op_b_imm
        && (instr.r_type.funct3 inside {3'b001}));
    }

endgroup : instr_cg
