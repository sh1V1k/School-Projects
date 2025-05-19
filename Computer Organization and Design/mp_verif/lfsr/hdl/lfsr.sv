module lfsr #(
    parameter logic [15:0]  SEED_VALUE = 'hECEB
) (
    input   logic           clk,
    input   logic           rst,
    input   logic           en,
    output  logic           rand_bit,
    output  logic   [15:0]  shift_reg
);

<<<<<<< HEAD
always @(posedge clk) begin
	if(rst)
		shift_reg <= SEED_VALUE;
	else begin
		if(en) begin
			rand_bit <= shift_reg[0];
			shift_reg <= {shift_reg[5] ^ (shift_reg[3] ^ (shift_reg[2] ^ shift_reg[0])), shift_reg[15:1]};
		end
	end
end
=======
    // TODO: Fill this out!
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

endmodule : lfsr
