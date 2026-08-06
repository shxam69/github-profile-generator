# Changelog

All notable changes to the **Particle Morph SVG Engine** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-06

### Added
- Rebranded repository as a generic **Particle Morph SVG Engine** for compiling custom interactive/animated particle SVGs.
- Created lightweight `main.py` entrypoint in the project root to trigger the compiler/renderer sequence.
- Added comprehensive architecture documentation suite under `docs/` covering each phase of the pipeline.
- Added GitHub repository templates and continuous integration test workflows.
- Migrated frame extraction debug scripts to `examples/extract_frames.py` with robust absolute path resolution.
- Added standard open-source assets: `LICENSE` (MIT), `CHANGELOG.md`, and `ROADMAP.md`.

### Removed
- Cleaned up empty placeholder files (`src/animation.py`, `src/logo_morph.py`).
- Removed obsolete backup scripts and duplicate modules (`compilerr.py`, `processorold.py`, `processorr.py`, `svg_rendererold.py`, `svg_rendererr.py`).
