# Roadmap

This document outlines the planned future direction for the **Particle Morph SVG Engine**.

## Phase 1: High-Performance Canvas Renderer
- [ ] Implement an alternative HTML5 Canvas renderer to support high frame rates (60 FPS) for larger particle counts.
- [ ] Add WebGL / WebGPU acceleration to render 100,000+ particles smoothly.

## Phase 2: Design Tool Integration & Live Preview
- [ ] Build a web-based UI editor allowing developers to import SVGs, upload portrait images, and preview animations interactively.
- [ ] Implement hot-reloading for parameters like unsharp masking, dithering thresholds, and transition times.
- [ ] Provide an interactive visualization of the particle path assignment (traveler routes).

## Phase 3: Developer Experience & Packaging
- [ ] Package the library and publish it to PyPI as `particle-morph-svg`.
- [ ] Expose a fully documented Command Line Interface (CLI) for parsing and rendering configurations directly from the terminal.
- [ ] Support export configurations for popular JS frontend frameworks (React, Vue, Svelte) to embed animations as component packages.
- [ ] Expand file format support beyond SVG (e.g. JSON timelines for Lottie players).
