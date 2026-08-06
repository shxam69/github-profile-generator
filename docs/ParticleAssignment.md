# Particle Assignment & Traveler Paths

The **Particle Assignment Engine** maps coordinates between dense portraits and sparse logo shapes using graph routing.

## Assignment Pipeline

```
+---------------------+        +--------------------+
|   dot_graph.json    |        |  logo_points.json  |
|  (14,248 vertices)  |        |    (900 points)    |
+---------------------+        +--------------------+
           │                              │
           └──────────────┬───────────────┘
                          ▼
            +---------------------------+
            |   KD-Tree & KNN Lookup    |
            |  - Locate nearest nodes   |
            +---------------------------+
                          │
                          ▼
            +---------------------------+
            |  Optimal Transport Solver |
            |  - Solve traveler path    |
            +---------------------------+
                          │
                          ▼
            +---------------------------+
            |   traveller_paths.json    |
            +---------------------------+
```

## Algorithms Used

1. **KD-Tree & KNN Querying**:
   Locates candidate portrait particles closest to logo boundaries, reducing the solution space.

2. **Optimal Path Solver (Travelers)**:
   Computes trajectories from portrait dot coordinate lists to target logo locations using minimum-cost matching, preventing particle paths from intersecting or tangling during transitions.
