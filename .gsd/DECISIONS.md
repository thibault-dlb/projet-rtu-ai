# DECISIONS.md

# Architecture & Design Decisions (ADR)

## ADR 001: GUI Framework - Dear PyGui
- **Status**: Accepted
- **Context**: Need a high-performance native UI for 60 FPS visualization.
- **Decision**: Use Dear PyGui (DPG).
- **Consequences**: Fast C++/GPU-backed rendering, but limited to desktop (Windows/Linux/MacOS).

## ADR 002: AI Core - neat-python & PyTorch
- **Status**: Accepted
- **Context**: Evolutionary system requirements.
- **Decision**: Use `neat-python` for topology evolution and `PyTorch` for GPU-accelerated tensor operations.
- **Consequences**: Robust evolution logic with flexible computation backend.
