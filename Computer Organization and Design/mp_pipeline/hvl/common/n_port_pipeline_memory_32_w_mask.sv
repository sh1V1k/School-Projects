module n_port_pipeline_memory_32_w_mask #(
    parameter CHANNELS = 2,
    parameter MAGIC = 1
)(
    mem_itf_w_mask.mem itf
);

    string memfile;
    initial begin
        $value$plusargs("MEMLST_ECE411=%s", memfile);
    end

    logic [31:0] internal_memory_array [logic [31:2]];

    logic [31:0] tag [CHANNELS][8];

    int delay_counter[CHANNELS], delay_counter_next[CHANNELS];
    bit stall[CHANNELS];
    bit tag_we[CHANNELS];
    bit mem_we[CHANNELS];

    logic [31:0] cached_addr   [CHANNELS];
    logic [3:0]  cached_rmask  [CHANNELS];
    logic [31:0] cached_rdata  [CHANNELS];
    logic [3:0]  cached_wmask  [CHANNELS];
    logic [31:0] cached_wdata  [CHANNELS];

    logic [31:0] current_write [CHANNELS];

    always_ff @(posedge itf.clk) begin
        for (int unsigned channel = 0; channel < CHANNELS; channel++) begin
            if (itf.rst) begin
                delay_counter[channel] <= '0;
                for (int i = 0; i < 8; i++) begin
                    tag[channel][i] <= '0;
                end
            end else begin
                delay_counter[channel] <= delay_counter_next[channel];
                if (tag_we[channel]) begin
                    tag[channel][cached_addr[channel][4:2]] <= cached_addr[channel];
                end
            end
        end
    end

    always_ff @(posedge itf.clk) begin
        for (int unsigned channel = 0; channel < CHANNELS; channel++) begin
            if (itf.rst || !stall[channel]) begin
                cached_addr [channel] <= itf.addr [channel];
                cached_rmask[channel] <= itf.rmask[channel];
                cached_wmask[channel] <= itf.wmask[channel];
                cached_wdata[channel] <= itf.wdata[channel];
                if (|itf.rmask[channel] || |itf.wmask[channel]) begin
                    cached_rdata[channel] <= (mem_we[channel] && (cached_addr[channel] == itf.addr[channel])) ? current_write[channel] : ($isunknown(itf.addr[channel]) ? 'x: internal_memory_array[itf.addr[channel][31:2]]);
                end
            end
        end
    end

    generate for (genvar channel = 0; channel < CHANNELS; channel++) begin : memory_logic
        always_comb begin
            itf.rdata[channel] = 'x;
            itf.resp[channel] = 1'b0;
            delay_counter_next[channel] = delay_counter[channel];
            stall[channel] = 1'b0;
            tag_we[channel] = 1'b0;
            mem_we[channel] = 1'b0;
            current_write[channel] = cached_rdata[channel];
            if (!itf.rst && (|cached_rmask[channel] || |cached_wmask[channel])) begin
                if (delay_counter[channel] != 0) begin
                    delay_counter_next[channel] = delay_counter[channel] - 1;
                    if (delay_counter[channel] == 1) begin
                        tag_we[channel] = 1'b1;
                    end
                    stall[channel] = 1'b1;
                end else begin
                    if (!MAGIC && (tag[channel][cached_addr[channel][4:2]] != cached_addr[channel])) begin
                        automatic int delay;
                        `ifndef ECE411_VERILATOR
                            std::randomize(delay) with {
                                delay dist {
                                    2 := 1,
                                    5 := 94,
                                    6 := 4,
                                    7 := 1
                                };
                            };
                        `else
                            delay = $urandom_range(0, 99);
                            case (delay) inside
                                0       : delay = 2;
                                [1:94]  : delay = 5;
                                [95:98] : delay = 6;
                                99      : delay = 7;
                            endcase
                        `endif
                        delay_counter_next[channel] = delay;
                        stall[channel] = 1'b1;
                    end else begin
                        if (|cached_rmask[channel]) begin
                            for (int i = 0; i < 4; i++) begin
                                if (cached_rmask[channel][i]) begin
                                    itf.rdata[channel][i*8+:8] = cached_rdata[channel][i*8+:8];
                                end
                            end
                            itf.resp[channel] = 1'b1;
                        end
                        if (|cached_wmask[channel]) begin
                            mem_we[channel] = 1'b1;
                            for (int i = 0; i < 4; i++) begin
                                if (cached_wmask[channel][i]) begin
                                    current_write[channel][i*8+:8] = cached_wdata[channel][i*8+:8];
                                end
                            end
                            itf.resp[channel] = 1'b1;
                        end
                    end
                end
            end
        end
    end endgenerate

    always_ff @(posedge itf.clk) begin
        if (itf.rst) begin
            internal_memory_array.delete();
            $readmemh(memfile, internal_memory_array);
            $display("using memory file %s", memfile);
        end else begin
            for (int unsigned channel = 0; channel < CHANNELS; channel++) begin
                if (mem_we[channel]) begin
                    internal_memory_array[cached_addr[channel][31:2]] = current_write[channel];
                end
            end
        end
    end

    always @(posedge itf.clk iff !itf.rst) begin
        for (int unsigned channel = 0; channel < CHANNELS; channel++) begin
            if ($isunknown(itf.rmask[channel]) || $isunknown(itf.wmask[channel])) begin
                $error("Memory Error: mask containes 'x");
                itf.error <= 1'b1;
            end
            if ((|itf.rmask[channel]) && (|itf.wmask[channel])) begin
                $error("Memory Error: simultaneous memory read and write");
                itf.error <= 1'b1;
            end
            if ((|itf.rmask[channel]) || (|itf.wmask[channel])) begin
                if ($isunknown(itf.addr[channel])) begin
                    $error("Memory Error: address contained 'x");
                    itf.error <= 1'b1;
                end
                if (itf.addr[channel][1:0] != 2'b00) begin
                    $error("Memory Error: address is not 32-bit aligned");
                    itf.error <= 1'b1;
                end
            end
        end
        // for (int unsigned i = 0; i < CHANNELS; i++) begin
        //     for (int unsigned j = i + 1; j < CHANNELS; j++) begin
        //         if (((|itf.rmask[i]) || (|itf.wmask[i])) && ((|itf.rmask[j]) || (|itf.wmask[j]))) begin
        //             if (itf.addr[i] == itf.addr[j]) begin
        //                 $error("Memory Error: same address simultaneously accessed on two ports");
        //                 itf.error <= 1'b1;
        //             end
        //         end
        //     end
        // end
    end

endmodule
