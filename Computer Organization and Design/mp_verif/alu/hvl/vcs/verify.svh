task verify_alu(output bit passed);
    bit [31:0] a_rand;
    bit [31:0] b_rand;
    bit [31:0] exp_f;

    passed = 1'b1;

    // TODO: Modify this code to cover all coverpoints in coverage.svh.
<<<<<<< HEAD
    for (int test = 0; test <= 500; ++test) begin
    for (int i = 0; i <= 6; ++i) begin
        std::randomize(a_rand);
        // TODO: Randomize b_rand using std::randomize.
	std::randomize(b_rand);
=======
    for (int i = 0; i <= 6; ++i) begin
        std::randomize(a_rand);
        // TODO: Randomize b_rand using std::randomize.
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

        // TODO: Call the sample_cg function with the right arguments.
        // This tells the covergroup about what stimulus you sent
        // to the DUT.

        // sample_cg(...);
<<<<<<< HEAD
	sample_cg(a_rand, b_rand, i[2:0]);
=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

        case (i)
            0: exp_f = a_rand & b_rand;
            1: exp_f = a_rand | b_rand;
<<<<<<< HEAD
            2: exp_f = ~a_rand;
            3: exp_f = a_rand + b_rand;
            4: exp_f = a_rand - b_rand;
            5: exp_f = a_rand << b_rand[4:0];
            6: exp_f = a_rand >> b_rand[4:0];
	    default: exp_f = 'x;
=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc
            // TODO: Fill out the rest of the operations.
        endcase

        // TODO: Drive the operand and op to DUT
        // Make sure you use non-blocking assignment (<=)

        // a <= a_rand;
        // b <= ...
        // aluop <= ...
        // valid_i <= 1'b1;
<<<<<<< HEAD
	a <= a_rand;
	b <= b_rand;
	aluop <= i[2:0];
	valid_i <= 1'b1;
=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

        // TODO: Wait one cycle for DUT to get the signal, then deassert valid

        // @(posedge clk)
        // valid_i <= ...
<<<<<<< HEAD
	@(posedge clk) begin
		valid_i <= 1'b0;
		end
=======
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

        // TODO: Wait for the valid_o signal to come out of the ALU
        // and check the result with the expected value,
        // modify the function output
        // "passed" if needed to tell top_tb if the ALU failed

        // @(posedge clk iff ...);
<<<<<<< HEAD
	@(posedge clk iff valid_o == 1'b1) begin
		if (exp_f != f) begin
				passed <= 1'b0;
			end else begin
				passed <= 1'b1;
			end
		end

    end
end
=======

    end
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

endtask : verify_alu
