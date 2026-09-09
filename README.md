# On-Device Video OCR Software

A pipeline that reads a digital numeric display (a thermometer, a scale, a multimeter, a counter — anything with a digital readout) from video footage using a local vision-language models, validates the readings, and exports one spreadsheet per video. Runs fully offline — no cloud calls, no API keys, no internet connection required at run time (though you'll probably need to be plugged into a wall).

I've used this to get readings from a standard LCD display, like one from a FLIR thermal camera. I haven't tested this with a 7-segment display, so your mileage may vary.

## AI Use Acknowledgement

I am not a programmer, I am a mechanical engineering student. I do not know python at this level and this was developed with Claude Code. Even though I designed the structure, vetted the plan files, and stayed involved during development, there may still be glitches that exist simply as a byproduct of AI-based programming. I had specific tasks that I needed to complete in the lab I work in and realized that there is an easier/better way to complete it, which it absolutely did by all means. The READMEs are also mostly AI generating, with my oversight, only because I didn't want to spend time on something that isn't really my professional focus, but still wanted to share what I had worked on.

## Why it exists

The aforementioned task was to go through videos second-by-second and record the data points in a spreadsheet; absolutely boring, tedious, and somewhat inaccurate. This pipeline automates the reading with a vision-language model, with safeguards and debugging tools that make it easy enough to manually read the frame the data point was derived from. Anything that looks wrong is **flagged for the user to check**, never corrected, smoothed, or dropped. I intentionally made sure to design such tools in, as minimal as they might be. The lab work required both accuracy and privacy, I didn't really care that some of the added steps reduced the speed, even though it was rather marginal.


## Versions

Each folder below is a **complete, independent build** of this tool — later versions are clean rebuilds, not incremental patches, so they don't share code with earlier ones. Pick the version you want to run and follow *its own* README for setup and usage.

| Version | Status | Notes |
|---|---|---|
| [`V9/`](./V9/README.md) | Current | Qwen3-VL-4B reader + MiniCPM-V judge, served via LM Studio; Gradio front-end. |

New versions get added the same way — as a new top-level folder with its own README — rather than as a new repository.

## Shared design principles

Every version in this repo follows the same non-negotiable rules, regardless of implementation:

- **Never silently alter a reading.** Validation flags readings for human review; it never corrects, smooths, interpolates, averages, or deletes one.
- **Never fabricate a confidence score.** Confidence is only reported when there's a genuinely independent, trustworthy signal for it (e.g. a second model's agreement) — otherwise it's `N/A`.
- **Fail fast and loud** on missing dependencies, unreachable models, or malformed input — no silent skips, no partial runs disguised as complete.
- **Explicit configuration.** Nothing about a reading is inferred from a filename or a hidden default; every setting that affects a run is visible and adjustable.

## License

Not yet decided — until a license file is added, all rights are reserved by the author.
