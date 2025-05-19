# ECE 411: mp_cache_sm GUIDE

## State-Machined 4-Way Set-Associative Cache

**This document, GUIDE.md, serves as a gentle, guided tour of the MP. For
strictly the specification and rubric, see [README.md](./README.md).**

# Cache Design

## Plan Ahead

It is **strongly recommended** that you carefully draw out the
datapath for the cache you will design.
This process will help you identify edge cases in your design and will
make writing the RTL significantly easier and faster. You should go to
office hours and ask course staff to look over your drawing as well.

## Architecture

If you're having trouble getting started, the course textbook [HP1]
details the design of a state-machined cache in "Chapter 5: Large
and Fast: Exploiting Memory Hierarchy".

## LRU Decoder Syntax

Within SystemVerilog, there exists a special type of case statement 
that allows you to match with wildcards. An example of this is shown below:

```systemverilog
unique casez (lru_raw)
    3'b0?0: lru_decode = 2'd0;
    // and more
endcase
```

In this case, `3'b0?0` will match both `3'b000` and `3'b010`. You may find 
such case statements useful when designing your LRU decoder, among other things.

# Verification

We have provided a skeleton testbench in `top_tb.sv` for testing
your cache. You will need to complete this testbench to verify that
your cache works. Similar to `mp_verif`, it has `TODOs` in the code
for you to complete. You should refer back to the testbenches in
`mp_verif` to get an idea of how to cleanly organize your testbench so
that you can exercise edge cases.

## Loading files into `simple_memory_256_wo_mask.sv`

The provided memory model loads a memory file using `$readmemh` from
a file specified on the Makefile command line. This file should be in
the same format as those you see in previous MPs. The specification
for this format can be found in IEEE 1800-2017 (the SystemVerilog specification)
Section 21.4. You should write your own memory file with contents of your choice.
Alternately, you can modify `simple_memory_256_wo_mask.sv` to have random data using
SystemVerilog's randomization features.

Let's say your memory file is located at `testcode/memory.lst`.
In this case, from the `sim` folder you would run:

```bash
make run_vcs_top_tb MEM=../testcode/memory.lst
```

## Constrained Random Testing
Caches present another nice application of constrained random vector
generation: generating addresses that exercise certain cases. For
instance, to verify your PLRU logic you may want to generate
addresses that have the same index bits, but have randomized tags and
offsets. An example of this would be as follows:

```systemverilog
std::randomize(addr) with {addr[8:5] == 4'hf;};
```

This will generate addresses that belong to set `15`. Note that you
don't generally need classes for constrained randomness, as you can 
also randomize inline using `std::randomize()`.

## Coverage

Setting up a simple covergroup can be extremely easy, and will help
pinpoint bugs such as not using a certain way or not entering a
certain state. You don't need to put the covergroup in a class,
and you can automate the sampling by triggering it at every clock,
like this:

```systemverilog
covergroup cg @(posedge clk);
    all_fsm_states           : coverpoint dut.control.state iff (!rst);
    writeback_to_pmem        : coverpoint dut.dfp_write {bins assert_write = {1};}
    read_all_cachelines_way0 : coverpoint dut.datapath.way0.addr iff (dut.ufp_rmask != '0);
endgroup : cg
```

Note the use of hierarchical references to make sampling easy. You can
extend this covergroup to track PLRU state, various state transitions,
and address space coverage.

# SRAM and OpenRAM

## What is SRAM?
In the past, to generate small memories, you have used a simple array
of flip-flops (for example, in the `mp_verif` and `mp_pipeline` CPU register files).
Such a design does not scale for large memories such as your cache data and
tag arrays. SRAMs offer better power, area, and timing outcomes for the design
compared to flip-flop based implementations via manual optimization
to provide the best density and timing on the provided technology.

The SRAM block is a hard IP. Hard IPs are generally IP blocks
that come as a package of both a simulation model and a physical
design file. The simulation model can be used during the RTL phase for
verification. Note that the simulation models are typically not
synthesizable, as they are merely Verilog files that mimic the behavior
of the IP. The physical design part will be integrated in a later step
of the design flow, where the layout is directly copied and pasted into
your final mask-level design.

## What is OpenRAM?
The tool we use to generate SRAM IPs is known as a memory
compiler. For ECE 411, we use the OpenRAM memory compiler.
The important files that will be generated include:
- The Verilog file used by VCS as a stub for simulation.
- The `.db` file used by DC for timing information and area
  estimation.
- The GDS file, which is the physical design file and is not used
  in this class.

To use OpenRAM, first prepare a config file in `sram/config`.
We have provided the SRAM config file you need to use for this MP.
Scripts in the provided file will convert them into the actual
configuration file used by OpenRAM.

Once the config file is prepared, navigate to the `sram` directory and run:

```bash
$ make
```

This will generate all relevant files in `sram/output`. The Makefile
also converts the timing model to a format that DC can use. This
timing model is used by the provided synthesis script.

Here is the list of signals for the SRAM blocks:
- `clk`: The clock.

- `csb`: Active low chip select, assert this when you need to read or
  write.

- `web`: Active low write enable, assert for writing and deassert for
  reading.

- `wmask`: Active high write enable for each byte. Will be ignored if
  `web` is deasserted.

- `addr`: The address.

- `din`: Write data.

- `dout`: Read data.

You can find the timing diagram in [README.md](./README.md).

Generic SRAMs like these do not have a deterministic power-on value, nor do
they have a reset pin. This is why we require you to use flip-flop based arrays
for the valid and PLRU arrays, as those need to be reset upon cache startup.

You may have noticed that the timing diagrams for the SRAMs contain
"old" values for write addresses. This is called a non-write-through
SRAM, or read-first SRAM. During a write, the read port will still
spit out some value. For non-write-through SRAM, this value is the old
value stored in the memory array. For write-through (or sometimes
called write-first) SRAM, this will be the newly written value.
You need to keep this non-write-though property in mind when designing
your cache.

A note about the SRAM Verilog model: we have heavily modified the
generator to produce modern Verilog in a style that is commonly used,
and changed some properties to fit our narrative. This is
a significant departure from OpenRAM's default behavior. If you are
unfortunate enough to need to use OpenRAM in the future after
this class, please keep this modification in mind.
