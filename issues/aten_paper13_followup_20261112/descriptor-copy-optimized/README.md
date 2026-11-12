# Generic descriptor-copy experiment

The runtime memory movement helper now orders traversal for contiguous output,
removes unit dimensions, collapses dimensions with compatible source and output
strides, decodes bounded 31-bit iteration counts with an exact precomputed
reciprocal and one correction, and copies aligned payloads with integer words.
Larger index counts retain the signed 64-bit decoder. Unaligned or arbitrary
payload sizes retain a byte path. The same output-disjointness proof guards
parallel stores; overlapping outputs and unsupported layouts retain the caller's
serial fallback. No arithmetic recognition or emitted library mapping changes.

The expanded GPU test passes 297,402,368 checked bytes, including signaling NaN
bits, signed zeros, offsets, input preservation, holes, reversed and broadcast
sources, permuted destinations, rank 16, non-power-of-two sizes, element sizes
1/2/3/4/5/8/12/16/24 bytes, and unaligned addresses. Four existing host tests pass. The queued-producer CUDA regression also passes
H2D/D2H/D2D copies, strided holes, overlapping views, and device/mapped/registered
host buffers with a pending real cuBLAS producer on a nonblocking stream.

Each benchmark retains frozen compiler, ABI, C oracle, and native ATen objects.
Only the Thrust runtime object changes; the timing harness is adjusted from its
historical twenty pairs to the requested five. All three pass unchanged strict
paper output/guard/input checks. Five warmups and five synchronized alternating
pairs give these raised runtime minima:

* Adaptive average pooling: 15.282596111 ms, previous first-five 15.989108086 ms.
  4.42% lower time; still 22.08x the paper runtime. Native: 0.491847992 ms.
* Pixel shuffle CPU backend: 25.622803926 ms, previous first-five 26.455898285 ms.
  3.15% lower time; still 79.93x the paper runtime. Native: 0.669548035 ms.
* Batch normalization: 49.887729168 ms, previous first-five 47.303535938 ms.
  5.46% higher time; still 189.98x the paper runtime. Native: 0.338534117 ms.

This does not resolve the large timing differences. Untimed adaptive-pooling
tracing shows only 0.262240 ms in the resident CUB operation, compared with more
than 15 ms in the full compiled invocation. Generated programs still allocate
and free large device intermediates each call. `slow-case-structure.json` records
the emitted mappings, copy counts, and allocation counts for sixteen slow cases;
allocation overhead is the next hypothesis to isolate.

`run_descriptor_copy.py` prepares the controlled relinks, runs the GPU unit test
first, then performs strict checks before any benchmark timing. Original binary,
object and source hashes, twenty-pair evidence, all five new samples, hardware
preflight, and unchanged dynamic-library hashes are preserved. These three paper
entries used twenty samples, so the new five-sample statistic is explicitly
different; both old all-twenty and old first-five minima remain recorded.
