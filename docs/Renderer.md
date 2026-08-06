# Renderer

The **Renderer** translates the compiled animation timeline database into a single vector graphic SVG file containing embedded SMIL animation tags (`<animate>`).

## Rendering Pipeline

```
+--------------------------+        +--------------------------+
|  compiled_timeline.json  |        |     dot_graph.json       |
+--------------------------+        +--------------------------+
             │                                    │
             └──────────────────┬─────────────────┘
                                ▼
                   +--------------------------+
                   |    Renderer Engine       |
                   |  - Instantiate SVG Node  |
                   |  - Parse active frames   |
                   +--------------------------+
                                │
                                ▼
                   +--------------------------+
                   |       SVGRenderer        |
                   |  - Serialize nodes       |
                   |  - Inject SMIL tags      |
                   +--------------------------+
                                │
                                ▼
                   +--------------------------+
                   |   animated_profile.svg   |
                   +--------------------------+
```

## Structure of the Generated SVG

- **Header / Viewbox**: Set by global config dimensions (e.g. `300x340`).
- **Def Blocks (`<defs>`)**: Pre-rendered templates, including styling variables, filters, gradients, and optional terminal graphics.
- **Circles (`<circle>`)**: Represents the active particles.
- **Animations (`<animate>`)**: Injected inside each `<circle>` tag, defining keytimes, attribute names, values, and begin times matching the compiled schedules.
