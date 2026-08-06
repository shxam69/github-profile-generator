# Performance Benchmarks & Profiles

This document presents performance metrics and memory profiles recorded from running the E2E validation pipeline.

## Execution Profile

The pipeline execution profile exhibits low memory usage and high CPU throughput, prioritizing vector computation over disk storage.

| Component / Phase | Avg. Exec Time (s) | Peak Memory (MB) | Output Size |
| :--- | :--- | :--- | :--- |
| Image Processing | ~0.15s | ~12 MB | PNG (~140KB) |
| Segmentation | ~0.08s | ~8 MB | Mask (~50KB) |
| Dithering & SVG | ~0.12s | ~15 MB | SVG (~11KB) |
| Dot & Graph Analysis | ~0.60s | ~45 MB | JSON (~1.2MB) |
| Intro & Drift Bands | ~0.45s | ~20 MB | JSON (~350KB) |
| Logo processing | ~0.85s | ~55 MB | JSON (~25KB) |
| Path Routing | ~1.10s | ~60 MB | JSON (~1.8MB) |
| Timeline Compiler | ~0.40s | ~30 MB | JSON (~6.0MB) |
| Multi-target Renderer | ~2.50s | ~120 MB | SVG (~30MB) |
| E2E Validator | ~1.40s | ~14 MB | JSON/MD (~1KB) |

## Memory Optimizations

1. **Iterative XML Parsing**:
   To parse the output 30MB animated SVG, the validator utilizes `xml.etree.ElementTree.iterparse` with manual node clearing, capping peak validator memory at **14 MB** instead of 300+ MB.
   
2. **KD-Tree Querying**:
   Replaces linear distance scans during KNN neighborhood searches, lowering complexity from \(O(N^2)\) to \(O(N \log N)\).
