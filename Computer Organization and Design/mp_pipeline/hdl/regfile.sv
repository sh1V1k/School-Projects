module regfile
(
    input   logic           clk,
    input   logic           rst,
    input   logic           regf_we,
    input   logic   [31:0]  rd_v,
    input   logic   [4:0]   rs1_s, rs2_s, rd_s,
    output  logic   [31:0]  rs1_v, rs2_v
);

    logic   [31:0]  data [31:0];

    always_ff @ (posedge clk) begin
        if (rst) begin
            for (integer i = 0; i < 32; i++) begin
                data[i] <= '0;
            end
        end else if (regf_we && (rd_s != 5'd0)) begin
            data[rd_s] <= rd_v;
        end
    end

    always_comb begin //changed to comb which is diff from reg file from mpverif
        if (rst) begin
            rs1_v = 'x;
            rs2_v = 'x;
        end else begin
            if(rs1_s != 5'd0) begin
                if(regf_we && rs1_s==rd_s) begin
                    rs1_v = rd_v;
                end else begin
                    rs1_v = data[rs1_s];
                end
            end else begin
                rs1_v = '0;
            end
            if(rs2_s != 5'd0) begin
                if(regf_we && ~(|(rs2_s ^ rd_s))) begin
                    rs2_v = rd_v;
                end else begin
                    rs2_v = data[rs2_s];
                end
            end
            else begin
                rs2_v = '0;
            end
        end
    end

endmodule : regfile
