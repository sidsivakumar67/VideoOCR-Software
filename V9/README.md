# Video OCR — V9

Reads a digital numeric display from video footage using a local vision-language model, validates the readings, and exports one spreadsheet per video. Runs entirely on-device, offline, through [LM Studio](https://lmstudio.ai/) — no cloud calls, no API keys.

It was built to read a thermometer display during tensile tests, but nothing about the pipeline is hard-wired to temperature: the expected reading format, valid value range, and the model's reading prompt are all configurable, so the same tool can read any digital numeric display (a scale, a multimeter, a counter, etc.).

> This folder is one version inside a repo that collects several independent builds of this tool. See the [repo root](../README.md) for the full list of versions. V9 is a clean, standalone rebuild — it does not share code with earlier versions.

## AI Use Acknowledgement

I am not a programmer, I am a mechanical engineering student. I do not know python and this was entirely developed with Claude Code. Even though I designed the structure, vetted the plan files, and stayed involved during development, there may still be glitches that exist simply as a byproduct of AI-based programming. I had specific tasks that I needed to complete in the lab I work in and realized that there is an easier/better way to complete it, which it absolutely did by all means. The READMEs are also mostly AI generating, with my oversight, only because I didn't want to spend time on something that isn't really my professional focus, but still wanted to share what I had worked on.

## Why it exists

The aforementioned task was to go through videos second-by-second and record the data points in a spreadsheet; absolutely boring, tedious, and somewhat inaccurate. This pipeline automates the reading with a vision-language model, with safeguards and debugging tools that make it easy enough to manually read the frame the data point was derived from. Anything that looks wrong is **flagged for the user to check**, never corrected, smoothed, or dropped. I intentionally made sure to design such tools in, as minimal as they might be. The lab work required both accuracy and privacy, I didn't really care that some of the added steps reduced the speed, even though it was rather marginal.

## How it works

1. **Sample** — pull one frame per second from the video (configurable rate), picking either the sharpest frame in each window (Laplacian variance) or simply the first frame of each window. No AI involved in this step.
2. **Filter** — optionally apply an image filter (greyscale, contrast/CLAHE, black & white, adaptive threshold, or none) to make the digits easier to read. Small frames are adaptively upscaled so the model gets enough pixels.
3. **Read** — send each filtered frame, one at a time and with no memory of prior frames, to a vision-language model running locally in LM Studio. Decoding is deterministic (greedy, temperature 0), so the same frame always reads the same way.
4. **Validate** — after every frame in the video has been read, run a post-pass of checks over the full set of readings: a format check, an optional physical-range check, a temporal-outlier check, and (optionally) a second, *different* vision-language model that independently re-reads every frame and flags disagreements. All of these only **flag** — none of them ever change a value.
5. **Export** — write one Excel workbook per video, with every reading, every check's result, and the reason for each flag. Frames whose readings were flagged are also copied out separately so you can jump straight to them.

## Requirements

- **Windows**, Python 3.10+ (developed and tested on Python 3.13).
- **[LM Studio](https://lmstudio.ai/)**, running locally with:
  - a vision-language reader model — developed against `qwen/qwen3-vl-4b` (Q4_K_M GGUF)
  - (optional) a *different* vision-language model to use as the independent judge — developed against `minicpm-v-4_5`. It must not be the same model as the reader, or it will share the reader's blind spots instead of catching them.
- A GPU capable of running a ~4B-parameter vision-language model at a usable speed is strongly recommended (I have a computer with an RTX 5070 Ti Laptop GPU, 12 GB VRAM). Any hardware compatible with LM Studio technically works, but it's just a question of how fast you want results.
- My Setup: I used a 2025 ROG Zephyrus G16 running Windows 11 with the U9-285H, 32GB RAM, and an RTX5070Ti 12GB Laptop GPU

## Installation

1. Install [LM Studio](https://lmstudio.ai/), then in it:
   - Download your reader model (e.g. `qwen/qwen3-vl-4b`) and, if you want the AI judge, a second, different vision-language model (e.g. `minicpm-v-4_5`).
   - Open the **Local Server** tab and start the server (default `http://127.0.0.1:1234`). Leave just-in-time model loading on so LM Studio can load/swap models automatically as the pipeline needs them.
   - If you have both an integrated and a dedicated GPU, make sure LM Studio is set to run on the dedicated one — the app's **Check GPU** button (see below) confirms this for you.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (or manually: `pip install opencv-python numpy pillow openpyxl gradio requests`)
3. Launch the app:
   ```bash
   python Modules/V9FrontendMod.py
   ```
   This opens a browser tab at `http://127.0.0.1:7860` with the UI (if something is already running on that port, it reuses it instead of erroring).

## Using it

1. **Upload a video.** Crop/trim it to the display in a video editor first — V9 does not do in-app cropping. Ensure the video is stable enough and has the desired framing.
2. **Check settings.** Sensible defaults are pre-filled (see the reference table below); every setting has a hover tooltip. At minimum, confirm the expected reading format matches your display (default is `NN.N`, e.g. a two-digit-plus-decimal thermometer reading).
3. **(Optional) Check GPU** — click **Check GPU (dGPU vs iGPU)** to confirm LM Studio is serving from your dedicated GPU rather than an integrated one.
4. **Run.** A progress bar and live run log track the current stage. If the AI judge is enabled and your LM Studio server needs a manual model swap (`judge_swap_mode = manual`), the run pauses with a **Continue** button once you've loaded the judge model; with the default `auto` mode LM Studio swaps models by itself.
5. **Cancel anytime** — you'll be offered **Discard** (throw away the partial run) or **Save partial results** (write out an Excel file for whatever was completed so far).
6. **Download** the resulting Excel file from the UI once the run finishes.

## Configuration reference

All settings are explicit — nothing is inferred from the video or filename. Front-end fields map directly to these defaults (`Modules/V9ConfigMod.py`):

| Setting | Default | Notes |
|---|---|---|
| Samples per second | `1.0` | Sampling rate along the video's timeline. |
| Sampling method | `laplacian` | `laplacian` picks the sharpest frame per window (variance of Laplacian); `first_frame` just takes the first frame of each window. |
| Filter | `greyscale` | One of `unfiltered`, `greyscale`, `contrast` (CLAHE), `black_and_white` (Otsu/fixed threshold), `adaptive` (adaptive threshold). All filters except `unfiltered` adaptively upscale small frames before reading. |
| Reading format mask | `NN.N` | Expected digit/decimal pattern used by the format check. |
| Allow negative | `false` | Whether a leading minus sign is valid. |
| Reading prompt | *(shown in UI)* | The exact system prompt sent to the reader/judge model. Editable, but changing it changes what the model returns. |
| LM Studio base URL | `http://127.0.0.1:1234/v1` | Where the local OpenAI-compatible server is expected. |
| Reader model id | `qwen/qwen3-vl-4b` | Must be loaded (or auto-loadable) in LM Studio. |
| Judge model id | `minicpm-v-4_5` | Only used if the AI judge is enabled; must be a different model from the reader. |
| Decoding | temperature `0.0`, top-k `1` | Fixed — greedy decoding, so a given frame always reads the same way. |
| Validation — master toggle | `on` | Turns the entire post-pass on/off. |
| Validation — format check | `on` | Flags readings that can't be normalized to the format mask. |
| Validation — range check | `off` | Off until you supply real physical min/max bounds — placeholder bounds would just generate false flags. |
| Validation — temporal check | `on` | Flags readings that deviate sharply from their neighbors in time (window `5`, cutoff `3.5`, modified z-score). Flags only — never smooths or removes. |
| Validation — AI judge | `on` | A second, different model independently re-reads every frame; disagreements are flagged. Doubles the read time for the video. |
| Judge swap mode | `auto` | `auto` relies on LM Studio's JIT loading to swap models between the reader and judge passes; `manual` pauses the run for you to swap models yourself in LM Studio. |
| Outputs directory | `Outputs` | Resolved relative to the project root regardless of where you launch from. |

## Output

Everything for a run lands in `Outputs/<video_name>/`:

- **`<video_name>.xlsx`** — one row per sampled second, columns:

  | Column | Meaning |
  |---|---|
  | `time_s` | Second in the video this row corresponds to. |
  | `frame` | Filename of the sampled frame (links the row to its image). |
  | `raw_reading` | Verbatim text the reader model returned. |
  | `judge_reading` | Verbatim text the judge model returned (blank if the judge is off). |
  | `parsed_value` | The reading normalized to the format mask, as a number. |
  | `format_check` / `range_check` / `temporal_check` / `judge` | Per-check result (blank if that check is disabled). |
  | `flagged` | `yes`/`no`. |
  | `flag_reason` | Which check(s) flagged the row, e.g. `temporal;judge`. |
  | `confidence` | The judge's agreement rate when the judge is enabled; `N/A` otherwise. The reader model never self-reports a confidence score — that would be a fabricated signal, not a measured one. |

- **`all_frames/`** — every sampled frame, already filtered (i.e. exactly what the model saw).
- **`flagged_frames/`** — copies of just the frames whose readings were flagged, for quick review.
- **`settings.txt`** — the exact settings used for the run, plus per-stage timing.

## Trust guarantees

These are non-negotiable in this codebase, not just defaults you can turn off:

- **A reading is never silently altered.** Validation only flags; it never corrects, smooths, interpolates, averages, or deletes a value. A statistical outlier might be real material behavior — a human decides that, not the code.
- **Confidence is never fabricated.** It's populated only from the AI judge's independent agreement. When the judge is off, confidence is `N/A` — never a guessed or model-self-reported number.
- **Every frame is read independently.** No conversation history, no prior frames or readings carry over between reads.
- **Failures are loud.** If LM Studio isn't running, a model isn't loaded, or the video/config is invalid, the run stops with a clear error — it never silently skips a frame or fakes a partial run as complete.

## Known limitations

- No in-app cropping — trim/crop the video to the display in a video editor before loading it.
- One video per run — no multi-video queue (batching multiple videos is out of scope for this version).
- Requires LM Studio running locally with the appropriate model(s) loaded; there's no bundled/portable model runtime yet.

## Project structure

```
V9/
  Modules/
    V9BackendMod.py       # LM Studio adapter — reads one image, returns a normalized result
    V9InterfacingMod.py   # orchestrator: video intake, sampling, filtering, output
    V9ValidationMod.py    # deterministic checks + AI judge
    V9FrontendMod.py      # Gradio UI
    V9ConfigMod.py        # explicit settings / defaults
  Outputs/                # created per run, one subfolder per video
  requirements.txt
  README.md               # this file
```
