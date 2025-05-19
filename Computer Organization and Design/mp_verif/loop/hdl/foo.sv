module foo(
    input   logic           clk,
    input   logic           rst,
    input   logic           req,
    input   logic   [3:0]   req_key,
    output  logic           ack
);

            logic   [3:0]   internal_counter;

    always_ff @(posedge clk) begin
        if (rst) begin
            internal_counter <= '1;
        end else begin
            internal_counter <= internal_counter - 4'd1;
<<<<<<< HEAD
            ack <= req && (req_key == internal_counter);
        end
    end

    // always_comb begin
    //     ack = req && (req_key == internal_counter);
    // end
=======
        end
    end

    always_comb begin
        ack = req && (req_key == internal_counter);
    end
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

endmodule
