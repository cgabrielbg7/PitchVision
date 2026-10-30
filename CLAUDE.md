# PitchVision

AI tactical analyst for football (soccer) from **broadcast-style video**: moving camera, zoom, cuts and replays. Output: player tracking on a top-down pitch, team stats, and an agent that answers tactical questions citing exact timestamps.

This is a **portfolio + learning project**. The goal is not only working software: I (Gabriel) must be able to explain every core module in a technical interview. Respect the ownership map and teaching mode below.

## Working agreement (read first)

- Respond in the language I write in (Spanish or English). Code, comments, commits and docs are always in English.
- **Plan before coding** any change that touches more than one file. Show the plan, wait for my OK.
- Small steps. One module or feature per task. Run tests after each change.
- Never add a dependency without asking and explaining why.
- Never commit videos, frames, model weights or anything under `data/`.
- When something is uncertain (an API, a library behavior), say so and check docs. Do not guess.
- When a design decision is made, add one line to `docs/decisions.md` (what, why, alternatives). Architecture decisions are mine: propose options with trade-offs, I choose.

## Teaching mode (my Python is rusty)

- When writing or reviewing Python, explain idioms I may not know (NumPy vectorization, broadcasting, comprehensions, dataclasses/pydantic, typing, context managers, generators).
- Prefer readable code over clever one-liners. When you vectorize with NumPy, show the loop version in a comment the first time.
- Never silently fix my code. Point to the issue, explain why, and let me fix it unless I say "fix it".
- If I'm stuck, give hints in steps: concept -> pseudo-code -> partial code -> full solution (only if I ask).
- After finishing a module, give me 3 interview-style questions about it.

## Ownership map: who writes what

The map evolves as my Python comes back.

| Level | Modules | Phase A (M1-M2) | Phase B (M3 onward) |
|---|---|---|---|
| **LEARN** | `stats/`, `field/homography.py`, `teams/`, `evals/` metrics, `cuda/` kernel | You give skeleton (signatures, docstrings, tests) + hints. I write the logic. | Same. Only review. |
| **PAIR** | `detect/`, `track/`, `shots/`, `field/keypoints.py`, `agent/`, `storage/`, `cli.py` | You may write a first version **with heavy comments**; I read it, modify it, and explain it back. | I write; you review and suggest. |
| **DELEGATE** | `web/` dashboard, `render/` overlays and plots, Dockerfile, CI, packaging, test boilerplate | You write it fully, keep it simple, summarize what you did. | Same. |

If I ask you to write a LEARN module, first ask: "Do you want hints first?"

## Architecture

```
Video MP4 -> frames (sampled, default 10 fps)
  -> shot classification: keep wide pitch views, drop replays / close-ups / crowd
  -> detect (YOLO: players, goalkeepers, referees, ball)
  -> track (ByteTrack) + re-identification across cuts (stretch goal)
  -> team assignment (jersey color clustering)
  -> pitch keypoint detection -> per-frame homography (pixels -> meters), temporally smoothed
  -> per-frame positions -> stats -> storage (Parquet/DuckDB)
                                       |
        agent (MCP tools over the data) / dashboard / overlay video + minimap
```

- **Heavy half (GPU, once per video):** everything up to storage.
- **Light half (CPU, instant):** agent, dashboard, reports. The agent NEVER reads video, only stored data.
- Frames are processed in memory and discarded. Only positions, stats and selected demo clips are persisted.
- Design every stage behind a small interface so a simpler fallback can be swapped in (e.g. manual 4-point calibration instead of keypoint homography).

## Repo layout

```
src/pitchvision/
  io/video.py            # frame extraction, fps sampling
  shots/classifier.py    # wide view vs replay/close-up (PAIR)
  detect/detector.py     # YOLO wrapper (PAIR)
  track/tracker.py       # ByteTrack, stable IDs, re-ID (PAIR)
  teams/assign.py        # jersey color clustering (LEARN)
  field/keypoints.py     # pitch landmark detection (PAIR)
  field/homography.py    # homography from keypoints + smoothing; manual fallback (LEARN)
  stats/metrics.py       # distance, speed, heatmaps, team shape, pressure (LEARN)
  storage/store.py       # Parquet + DuckDB (PAIR)
  agent/tools.py         # tool functions over stored data (PAIR)
  agent/mcp_server.py    # MCP server exposing tools (PAIR)
  agent/agent.py         # agent loop (Claude Agent SDK), timestamp citations required (PAIR)
  render/overlay.py      # annotated video + minimap (DELEGATE)
  cli.py                 # `pitchvision process|ask|report` (PAIR)
evals/
  vision/                # detection, tracking, shot classification, homography error
  agent/                 # questions with computed answers + grounding checks
  perf/                  # fps/latency: PyTorch vs ONNX vs TensorRT FP16/INT8
cuda/                    # custom CUDA kernel + benchmark vs NumPy (LEARN)
notebooks/               # exploration only, never the source of truth
web/                     # Next.js dashboard (DELEGATE)
docs/decisions.md        # design decision log
data/                    # gitignored: raw/, processed/, labels/, SOURCES.md
tests/
```

## Stack

- Python 3.12, `uv`, `ruff`, `pytest`, type hints everywhere, `pydantic`, `typer`.
- Vision: `ultralytics` (YOLO, YOLO-pose for pitch keypoints, ByteTrack; AGPL-3.0, fine for this open-source repo), `opencv-python`, `numpy`, `scikit-learn`.
- Data: `pyarrow` / Parquet, `duckdb`.
- Agent: `claude-agent-sdk`, `mcp` (Python SDK). Optional comparison model via NVIDIA NIM or a local model.
- GPU: local NVIDIA GPU (check with `nvidia-smi`), CUDA, TensorRT (via Ultralytics export), custom kernel with CUDA C++ or Numba/CuPy.
- Web: Next.js + TypeScript, reads exported JSON/Parquet through a small FastAPI endpoint.

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format .
uv run pytest
uv run pitchvision process data/raw/clip.mp4 --fps 10 --out data/processed/clip/
uv run pitchvision ask data/processed/clip/ "Who covered the most distance?"
uv run python -m evals.run --suite agent   # suites: vision | agent | perf
nvidia-smi
```

## Conventions

- Pure functions for math (homography, stats) so they are easy to unit test with synthetic inputs.
- Units: positions in meters on a 105 x 68 pitch, time in seconds from video start. Name variables with units (`speed_mps`, `t_s`).
- Every agent answer must cite timestamps that exist in the stored data. No timestamp = invalid answer.
- Log, do not print. Config via a pydantic settings object, never hardcoded paths.
- Keep files under ~300 lines. Prefer clarity over cleverness.

## Evals (the core of the project)

1. **Vision:** detection precision/recall, tracking quality, shot classification accuracy, and homography reprojection error on labeled frames.
2. **Performance:** fps and latency per backend, plus the accuracy lost at FP16/INT8.
3. **Agent correctness:** questions whose answer is computed directly from the data (exact match).
4. **Grounding:** % of claims in a report backed by a real timestamp.
5. **Cost/latency** per question and per model.

Rules: deterministic where possible, results saved to `evals/results/` as JSON with model, date and git commit. Report counts, not just percentages. CI runs the fast suites on every PR.

## Data and legal

- Only use footage I own, have written permission for, or whose license allows reuse (e.g. CC BY, with attribution). Record source, license and permission for every file in `data/SOURCES.md`.
- "No copyright intended" in a video description does not grant any rights. Prefer getting original files directly from the owner (e.g. an amateur or semi-pro league).
- Never commit footage. Public demo clips only when the license or permission covers publishing.

## Milestones (~8 weeks)

- [ ] M0: Python refresh on project topics (NumPy, OpenCV, pytest, uv)
- [ ] M1: detection + tracking on a broadcast clip, positions saved in pixel space
- [ ] M2: shot classification + team assignment
- [ ] M3: pitch keypoints + per-frame homography + stats in meters (manual calibration fallback)
- [ ] M4: MCP server + agent with timestamp citations
- [ ] M5: eval suites + CI -> **v1, goes on CV**
- [ ] M6: TensorRT FP16/INT8 + custom CUDA kernel + benchmarks
- [ ] M7: README (architecture, results, limitations, "How I used Claude Code"), demo video, LinkedIn post

## Definition of done (per milestone)

Tests pass, ruff clean, decision log updated, README section updated, and I can explain the code without notes.
