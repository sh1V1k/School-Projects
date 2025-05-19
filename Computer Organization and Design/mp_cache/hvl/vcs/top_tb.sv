module top_tb;
    //---------------------------------------------------------------------------------
    // Waveform generation.
    //---------------------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("dump.fsdb");
        $fsdbDumpvars(0, "+all");
    end

    //---------------------------------------------------------------------------------
    // TODO: Declare cache port signals:
    //---------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------
    // TODO: Instantiate the DUT:
    //---------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------
    // TODO: Generate a clock:
    //---------------------------------------------------------------------------------


    //---------------------------------------------------------------------------------
    // TODO: Verification constructs (recommended)
    //---------------------------------------------------------------------------------
    // Here's ASCII art of how the recommended testbench works:
    //                                +--------------+                           +-----------+
    //                       +------->| Golden model |---output_transaction_t--->|           |
    //                       |        +--------------+                           |           |
    //  input_transaction ---+                                                   | Check ==  |
    //                       |        +------+                                   |           |
    //                       +------->|  DUT |---output_transaction_t----------->|           |
    //                                +------+                                   +-----------+

    // Struct that defines an "input transaction" -- this is basically one
    // operation that's done on the cache.
    typedef struct packed {
        logic [31:0] address; // Address to read from.
        bit transaction_type; // Read or write? You could make an enum for this.
        // ... what else defines an input transaction? Think: rmask/wmask, data...

        // Note that it's useful to include the DFP signals here as well for
        // planned misses, like this:
        bit [255:0] dfp_rdata;
        // What else?
    } input_transaction_t;

    // The output transaction tells us how the cache is expected to behave due
    // to an input transaction.
    typedef struct packed {
        bit caused_writeback;
        bit caused_allocate;
        bit [31:0] returned_data;
        bit [255:0] dfp_writeback_data;
        // what else do you need?
    } output_transaction_t;

    logic [255:0] data_golden_arrays[4];
    // Similarly, make arrays for tags, valid, dirty, plru.

    function input_transaction_t generate_input_transaction();
        // This function generates an input transaction. 

        input_transaction_t inp;

        // Pick whether to generate a hit or miss.
        bit do_hit;
        std::randomize(do_hit);

        if (do_hit) begin
            // If we're generating a hit, we need to request an address that's
            // actually in the cache. Call:

            // get_cached_addresses(); Write this function to query golden tag
            // arrays, then fill out inp.address and other inp fields.
        end else begin // do miss
            // do:
            // std::randomize(inp) with {...};
            // Since it's a miss, we must fill out inp.dfp_* signals.
            // inp.address can be completely random.
        end
    endfunction : generate_input_transaction

    function output_transaction_t golden_cache_do(input_transaction_t inp);
        output_transaction_t out;
        // Do operations on the arrays, and fill up "out" correctly. Use "="
        // assignment here: this is not RTL. It is a behavioral software model 
        // of the cache.
    endfunction : golden_cache_do

    task drive_dut(input input_transaction_t inp, output output_transaction_t out);
        // Do inp operation on the DUT by driving with "<=".
        // Fill out an output_transaction_t struct while doing so depending on
        // what the DUT does. Refer to mp_verif to see how to drive and
        // monitor DUT signals. It may be useful to use fork...join, or to
        // also take as input the golden model's output struct to make it
        // easier to drive.
    endtask : drive_dut

    function compare_outputs(output_transaction_t golden_out, output_transaction_t dut_out);
        // Compare the output structs, and $error() if there's a mismatch.
    endfunction : compare_outputs

    //---------------------------------------------------------------------------------
    // TODO: Main initial block that calls your tasks, then calls $finish
    //---------------------------------------------------------------------------------

endmodule : top_tb
