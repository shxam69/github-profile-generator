# Logo Processor

The **Logo Processor** parses vector SVG files and samples them into a fixed particle budget utilizing a feature-aware probability weight map.

## Sampling Pipeline

```
+------------------+
|    Input SVG     |
+------------------+
         │
         ▼
+------------------+
|  Rasterization   |  (Coarse scale -> High-Resolution binary mask)
+------------------+
         │
         ▼
+------------------+
|  Feature Extraction
|  - Canny Edges
|  - Harris Corners
|  - Curvature Map
|  - Dist Transform
+------------------+
         │
         ▼
+------------------+
| Probability Map  |  (Renormalizes feature-zone vs. interior-zone weights)
+------------------+
         │
         ▼
+------------------+
|  Random Choice   |  (Weighted sampling based on probability map)
+------------------+
         │
         ▼
+------------------+
| logo_points.json |  (900 coordinate pairs)
+------------------+
```

## Feature Mapping Metrics

- **Edges**: Extracts symbol outlines using Canny Edge detection.
- **Corners**: Boosts sharp angles using the Harris Corner detector.
- **Curvature**: Computes turning angles on vector contours, isolating wing tips, claws, or detailed tails.
- **Interior Fill**: Distributes a fallback weight budget (typically 25%) uniformly inside the mask to prevent hollow layouts.
