# Timeline Compiler

The **Timeline Compiler** resolves multiple animation groups, keyframes, and trajectories into a flat, chronological sequence of events.

## Key Concepts

- **AnimationKeyframe**: Defines a state at a specific relative timestamp (e.g. `time = 1.5`, properties = `{"opacity": 0.5, "cx": 100.2}`).
- **AnimationGroup**: A list of targets (dot/particle IDs) sharing a keyframe sequence with common start offsets, durations, repetitions, and easing functions.
- **AnimationTimeline**: A master container holding several `AnimationGroups` and orchestrating execution.

## Compilation Pipeline

The Compiler iterates through all scheduled timelines, resolves relative group offsets into global absolute times, and flattens properties per particle ID.

```
+--------------------------------------------------------+
|                      Input JSONs                       |
| (Intro Shimmers, Drift Bands, Traveller Logo Paths)   |
+--------------------------------------------------------+
                           │
                           ▼
+--------------------------------------------------------+
|                 AnimationCompiler                      |
|  - Parse active components                             |
|  - Convert relative coordinates to target offsets     |
|  - Resolve absolute start/end timestamps               |
+--------------------------------------------------------+
                           │
                           ▼
+--------------------------------------------------------+
|                compiled_timeline.json                  |
|  - Chronologically sorted events list                  |
|  - Affected target IDs and coordinate attributes      |
+--------------------------------------------------------+
```

## Easing Functions

The compiler supports:
- `linear`: Standard linear transition.
- `ease-in-out`: Smooth acceleration and deceleration.
- `shimmer-intro`: Customized sine-wave intensity fluctuations for intro waves.
