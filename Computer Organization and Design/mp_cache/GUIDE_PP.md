# ECE 411: mp_cache_pp GUIDE

## Pipelined 4-Way Set-Associative Cache

**This document highlights the differences in detail for
the pipelined version of this cache. For strictly the specifications, 
see [README_PP.md](./README_PP.md)**

# Cache Design

## Architecture

In a state-machined design the entire cache serves one request at a time,
which is the primary source of inefficiency. All control signals come from the explicit
state machine which dictates the current status of the cache.

![pipeline_stage](./doc/images/pipeline_stage.svg)

For the pipelined version, while the current request has been registered on
the pipeline register (including the register in SRAM) and been processed,
another request is already lined up on the input of the pipeline register. This way,
you can achieve the ideal throughput of 1 request serviced per cycle. There is still 
"state" in this pipelined cache, but it is now implicitly encoded and control signals 
will come from performing logic on these states. Say, for example, the condition for 
writing back is "if the current status on the right hand side stage says it is a miss 
and dirty". While waiting for the DFP to finish, you would of course want to stall the 
pipeline. Once the writeback is done, you can now change the "state" by marking this line 
as clean. The combinational logic will then realize that this is a clean miss, after which
the cache will continue to stall and start fetching the line. After the DFP response, you 
will update the "state" by writing the new line into the data array and updating the tag, valid 
and dirty bits. On the next cycle, everything will now look like a hit.
