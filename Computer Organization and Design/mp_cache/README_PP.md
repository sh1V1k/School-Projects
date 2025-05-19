# ECE 411: mp_cache_pp README

## Pipelined 4-Way Set-Associative Cache

**This document highlights the differences in specifications for
the pipelined version of this cache. For a more comprehensive summary 
of this version, see [GUIDE_PP.md](./GUIDE_PP.md)**

# Design Specifications

- 2-stage pipelined:
  - 1 cycle latency in response on cache hits
  - 1 access per cycle throughput on cache hits

# Cache Timing Requirements

The cache must obey the following timing requirements:

## Hits

<p align="center"> <img src="doc/images/cache_pp_read_hit.svg"/> <p
  align="center">Read hit timing diagram</p> </p>

<p align="center"> <img src="doc/images/cache_pp_write_hit.svg"/> <p
  align="center">Write hit timing diagram</p> </p>

<p align="center"> <img src="doc/images/cache_pp_mixed_hit.svg"/> <p
  align="center">Mixed hit timing diagram</p> </p>

## Clean Misses

<p align="center"> <img src="doc/images/cache_pp_read_miss_clean.svg"/> <p
  align="center">Read with clean miss timing diagram</p> </p>

<p align="center"> <img src="doc/images/cache_pp_write_miss_clean.svg"/> <p
  align="center">Write with clean miss timing diagram</p> </p>

## Dirty Misses

<p align="center"> <img src="doc/images/cache_pp_read_miss_dirty.svg"/> <p
  align="center">Read with dirty miss timing diagram</p> </p>

<p align="center"> <img src="doc/images/cache_pp_write_miss_dirty.svg"/> <p
  align="center">Write with dirty miss timing diagram</p> </p>

# Grading

## Submission
You will be graded on the files on the `cache_pp` branch in your class GitHub repository on the specified deadline.
