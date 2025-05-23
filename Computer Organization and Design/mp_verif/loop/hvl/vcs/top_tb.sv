module top_tb;

    //----------------------------------------------------------------------
    // Waveforms.
    //----------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("dump.fsdb");
        if ($test$plusargs("NO_DUMP_ALL_ECE411")) begin
            $fsdbDumpvars(0, dut, "+all");
            $fsdbDumpoff();
        end else begin
            $fsdbDumpvars(0, "+all");
        end
    end

    //----------------------------------------------------------------------
    // Generate the clock.
    //----------------------------------------------------------------------
    bit clk;
    always #500ps clk = ~clk;

    //----------------------------------------------------------------------
    // Generate the reset.
    //----------------------------------------------------------------------
    bit rst;
    task do_reset();
        rst = 1'b1;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
    endtask : do_reset

    //----------------------------------------------------------------------
    // DUT instance.
    //----------------------------------------------------------------------

    logic ack;

    loop dut(.*);

    //----------------------------------------------------------------------
    // Verification tasks/functions
    //----------------------------------------------------------------------
    task verify_loop();
        @(posedge clk);

        repeat (100) begin
            repeat (15) @(posedge clk);
            if (!ack) begin
                $error("TB Error: Verification Failed");
                $fatal;
            end
        end
    endtask : verify_loop

    //----------------------------------------------------------------------
    // Main process.
    //----------------------------------------------------------------------
    initial begin
        do_reset();
        verify_loop();
        $finish;
    end

    //----------------------------------------------------------------------
    // Timeout.
    //----------------------------------------------------------------------
    initial begin
        #50us;
        $error("TB Error: Timed out");
        $fatal;
    end

endmodule
