    longint timeout;
    initial begin
        $value$plusargs("TIMEOUT_ECE411=%d", timeout);
    end

    mem_itf_w_mask #(.CHANNELS(2)) mem_itf(.*);
<<<<<<< HEAD
    n_port_pipeline_memory_32_w_mask #(.CHANNELS(2), .MAGIC(1)) mem(.itf(mem_itf));

=======
    n_port_pipeline_memory_32_w_mask #(.CHANNELS(2), .MAGIC(0)) mem(.itf(mem_itf));
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

    mon_itf mon_itf(.*);
    monitor monitor(.itf(mon_itf));

    cpu dut(
        .clk            (clk),
        .rst            (rst),

        .imem_addr      (mem_itf.addr [0]),
        .imem_rmask     (mem_itf.rmask[0]),
        .imem_rdata     (mem_itf.rdata[0]),
        .imem_resp      (mem_itf.resp [0]),

        .dmem_addr      (mem_itf.addr [1]),
        .dmem_rmask     (mem_itf.rmask[1]),
        .dmem_wmask     (mem_itf.wmask[1]),
        .dmem_rdata     (mem_itf.rdata[1]),
        .dmem_wdata     (mem_itf.wdata[1]),
        .dmem_resp      (mem_itf.resp [1])
    );

    assign mem_itf.wmask[0] = '0;
    assign mem_itf.wdata[0] = 'x;

    `include "rvfi_reference.svh"

    always @(posedge clk) begin
        if (mon_itf.halt) begin
            $finish;
        end
        if (timeout == 0) begin
            $error("TB Error: Timed out");
            $fatal;
        end
        if (mem_itf.error != 0 || mon_itf.error != 0) begin
            $fatal;
        end
        timeout <= timeout - 1;
    end
