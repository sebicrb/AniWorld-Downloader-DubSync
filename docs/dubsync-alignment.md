# DubSync: How the Automatic Audio Alignment Works

DubSync grafts a web-sourced German dub onto an archive-quality video file
(e.g. a Blu-ray rip that only carries JP/EN audio). Before the dub can be
muxed in, its timing has to match the target video — different releases trim
intros differently, add black frames, or run at a different speed entirely.
This document explains how the aligner
([`align.py`](../src/aniworld/models/aniworld_to/dubsync/align.py)) detects
those differences automatically.

## The problem

The obvious approach — correlate the two audio tracks directly — cannot work:
the dialogue is *different speech in different languages*. What both tracks
share is the **music and sound-effects bed**: a door slam, an explosion, or a
musical sting happens at the same story moment in every language version of an
episode. The aligner is built entirely around locking onto that shared signal.

Two kinds of misalignment occur in practice:

| Kind | Cause | Fix |
| --- | --- | --- |
| **Constant offset** | Different intro/pre-roll trimming, black frames | Delay the dub by a fixed amount (`-itsoffset`, lossless) |
| **Linear drift** | PAL-sourced dubs run 25/23.976 ≈ 4.3 % fast, so the offset grows over the episode | Retime the dub (`atempo`, re-encodes the dub track only — opt-in) |

## Step 1 — Onset envelopes

Both tracks are decoded to mono 8 kHz PCM with ffmpeg. For the target video,
the Japanese audio stream is preferred as the reference (falling back to the
first audio stream) — though the method does not depend on the language, since
the signal being matched is the non-dialogue bed.

Each track is then collapsed into an **onset envelope**:

1. Chop the samples into 10 ms frames (100 frames per second).
2. Reduce each frame to its RMS energy, log-compressed: `log(1 + 100·rms)`.
3. Take the frame-to-frame *difference* and clip negative values to zero
   (half-wave rectification), keeping only "it suddenly got louder" events.
4. Z-normalise the result.

The output is a spiky 100 Hz signal dominated by loud transients — music hits
and SFX — while continuous speech contributes only small ripples. That is what
makes the envelope nearly identical between the dub and the original even
though every spoken word differs.

## Step 2 — FFT cross-correlation

Finding the constant offset means asking: *if the dub's envelope is slid by
`X` seconds, how well do its spikes line up with the reference's spikes?* —
for every possible `X`. Computed naively this is O(n²); via the correlation
theorem it is two FFTs and a multiply:

```
corr = irfft( rfft(ref_env) · conj(rfft(dub_env)) )
```

`corr[k]` is the match score for delaying the dub by `k` envelope samples
(negative lags wrap around and mean "advance the dub"). The search is bounded
to ±600 s by default. The best lag is refined below the 10 ms envelope
resolution by parabolic interpolation over the peak and its two neighbours.

For a 24-minute episode the envelopes are ~145 000 samples, so the whole
correlation takes milliseconds.

## Step 3 — Confidence as a z-score

The raw peak height is meaningless on its own, so it is converted into a
z-score: how many standard deviations the peak rises above the rest of the
searched correlation range (excluding ±1 s around the peak itself).

* A genuine match is a needle: z ≫ 10, often in the hundreds.
* Two unrelated tracks produce a mushy landscape whose tallest bump is only
  z ≈ 3–5.

The default acceptance threshold is **z ≥ 8** (`DEFAULT_MIN_CONFIDENCE`).
Below it, the pipeline does not trust the detection: the episode falls back to
offset 0 (or the user's manual `--dubsync-offset`) and is flagged
*low-confidence* in the report so it can be verified by ear.

## Step 4 — Drift detection by tempo scanning

A speed-changed dub breaks the constant-offset model *and* sabotages the
correlation itself: each music hit wants a slightly different lag, so the peak
smears out (a 4 % PAL drift spreads it over ±30 s and the confidence collapses
into the noise floor).

The scan turns that symptom into the detector. The dub envelope is
time-stretched by candidate speed factors and re-correlated at each one:

* At the **wrong** tempo the peak stays smeared and flat.
* At the **true** tempo every hit snaps to the same lag and the peak becomes
  dramatically sharp.

The peak is needle-thin in tempo: over a 50-minute episode a factor just
7·10⁻⁴ off still smears it by two seconds. A coarse grid therefore steps
straight over the true ratio on long episodes (this is what failed on a
live-action season: z ≈ 4 on every episode, everything fell back to offset 0).
So the scan runs in two stages:

1. **Standard ratios first.** The speeds real dubs actually differ by —
   25/23.976 and 25/24 (PAL speed-up), 24/23.976 (NTSC slip), and their
   inverses — are each probed with a ±10⁻⁴ neighbourhood at 2·10⁻⁵ steps.
   If one clears the guard rails below it is polished to ~10⁻⁶ and wins.
2. **Generic grid otherwise.** Hierarchical over `1 ± 6 %`: a coarse grid,
   then finer grids around the running best. When the identity peak is
   already confident the coarse step is 2·10⁻³ (cheap; the true factor is
   near 1 anyway); when it is not, the coarse step shrinks to
   0.3 s ÷ duration so the true factor can't be stepped over.

Each correlation reuses a precomputed reference FFT.

Guard rails, in order:

1. Only attempted when the dub is ≥ 5 minutes (`_DRIFT_MIN_DURATION`) —
   shorter material cannot distinguish drift from noise.
2. A non-identity tempo must clear the normal confidence threshold **and**
   beat the identity-tempo peak by a margin of 5 z (`_TEMPO_MARGIN`).
   Random noise cannot fake that, because real drift *un-smears* the peak
   rather than just nudging it.
3. The resulting drift must exceed 0.2 s across the episode
   (`DRIFT_THRESHOLD`) before it is reported at all.

When drift is confirmed, the result carries the `atempo` factor
(`tempo_ratio = 1/detected speed`) and the residual constant offset measured
at the winning tempo (`post_tempo_offset`).

## Step 5 — Act-break cuts (piecewise alignment)

TV versions of a show are often edited: the German broadcast cut drops a few
seconds at every act break (typically the fade to black before an ad break).
After tempo correction, the offset then *steps* through the episode — on
*Animal Kingdom* S02E01: +1.03 s → +3.72 s → +5.94 s → +8.61 s → +10.97 s,
one jump per act. No single offset fits; any constant one leaves most of the
episode seconds out of sync.

`detect_offset(..., find_cuts=True)` builds a piecewise map:

1. **Chunks.** The (tempo-corrected) dub envelope is cut into 60 s chunks,
   each correlated on its own within ±90 s of the global offset.
2. **Runs.** Consecutive confident chunks (z ≥ 8) whose shift agrees within
   0.12 s form a run. A lone chunk that disagrees is dropped unless its peak
   is very sharp (z ≥ 12) — real acts span several chunks. Runs closer than
   0.15 s are merged.
3. **Boundaries.** Between two runs, 20 s sub-windows (2 s hop) are labelled
   "old shift" or "new shift" when their peak is sharp. Periodic music
   correlates at *both* shifts but only bluntly, so it stays unlabelled
   instead of dragging the cut (a summed-correlation score alone was fooled
   by exactly this, by ~70 s). The split that agrees with the most labels
   bounds the cut to an interval (softened by 5 s).
4. **Snapping.** Inside the interval, the cut goes where the patched span is
   quietest: if the dub is missing material, the stretch of the *reference*
   that will play against padded silence — i.e. the fade the TV cut removed;
   if the dub has extra, the stretch of the dub that gets dropped.

The result is a list of `Segment(start, end, shift)` in post-tempo dub time.

## Step 6 — Applying the result

* **Constant offset only** (the common case): the dub is muxed with
  `-itsoffset <offset>` and `-c copy` — fully lossless, the dub stream stays
  bit-exact.
* **Drift and/or cuts detected**: `render_dub` re-renders the dub in one
  ffmpeg pass — each segment is its own seeked input (splitting one decoded
  stream would buffer the whole episode in memory), `atempo` corrects the
  speed, `apad` inserts silence where the dub is missing material (or the
  next segment's head is trimmed where it has extra), and `concat` joins
  them. The first segment's shift stays the constant `-itsoffset`. This
  decodes and re-encodes the dub (AAC 192k: the source is already a lossy
  stream rip), so it only runs when the user opts in
  (`--dubsync-allow-resample`, "Fix wrong speed and cut scenes" in the UI).
  Video and all original audio remain untouched either way. Without the
  opt-in, the constant offset is used and the job is flagged in the queue
  that sync will wander.

Measured on *Animal Kingdom* S02E01 (15 s windows, 5 s hop, ±0.1 s
tolerance): the old constant-offset graft was in sync in 1 of 571 windows;
the re-rendered dub in 344, with the rest too quiet to judge or isolated
borderline peaks — none clustered at the cuts.

`aniworld <url> --dubsync-target <dir> --dubsync-dry-run` runs match +
alignment for every episode and prints the detected offsets, confidences and
drift **without writing to any target file** — the recommended way to build
trust in the aligner before it touches archive files.

## Reuse: multi-language downloads

The same detector runs outside DubSync. Downloading an episode you already have
in another language merges the new stream into the existing file, and the
streams come from different hoster encodes that rarely start at the same
instant — so the merge asks `detect_offset` for the shift first instead of
concatenating blindly (which is what used to leave the second track playing
late).

* An extra **dub** becomes another audio track, muxed through the same
  `-itsoffset` + `-c copy` path DubSync uses.
* A hardsubbed **"Sub" variant** brings its own video, so its video and audio
  are appended as one unit shifted by a single offset — they are already in
  sync with each other, only with the file as a whole.

Both cases mark the appended streams non-default so players keep the original
primary, and both fall back to an unshifted merge when the detection lands
below `DEFAULT_MIN_CONFIDENCE` — the alignment is an improvement on the old
behaviour, never a precondition for the download succeeding.
`--no-merge-align` (or `ANIWORLD_MERGE_ALIGN=0`) opts out entirely.

The search is bounded to ±60 s here rather than the ±600 s default: two
encodes of the same episode start within seconds of each other, and a wider
window only lets a noise peak minutes away win.

### Speed mismatches are retimed, never shifted

Drift is not a corner case on this path — it is the *norm* for German TV dubs,
which are PAL (25 fps) while the original-language encode is film rate
(23.976), so the two run ~4.3 % apart. Observed live on Family Guy S01E01:
the identity-tempo peak was noise (z=4.1, claiming a −13 s offset) while the
tempo scan locked onto exactly 0.95904 at z=19.5.

That combination is a trap, because `detect_offset` overwrites
`result.confidence` with the *drift* confidence while `result.offset` keeps
the identity-tempo value. Reading the two together says "high confidence,
apply this offset" about a number that was measured at noise level — which is
how a merge once placed a track 95 s late. **When `has_drift` is set, the
constant offset must be ignored**; only `post_tempo_offset` is meaningful, and
only after the audio has actually been retimed.

So the download path retimes rather than shifts: `resample_dub` applies the
`atempo` factor to the added track (re-encoded to AAC here, not FLAC, since
the source is an already-lossy stream rip and a lossless track would dwarf the
video), then the residual `post_tempo_offset` is applied as usual. Only the
freshly added track is ever re-encoded. `--no-merge-resample` disables it, and
a **"Sub" variant is never retimed at all** — its audio arrived with its own
video, so speeding up one without the other would desync them; that case is
merged unshifted with a warning instead.

### Player support for the appended streams

Extra **audio** tracks are understood everywhere. An extra **video** track (a
hardsubbed "Sub" variant) is not: mpv and VLC switch between them fine, but
media servers generally play only the first video track — Jellyfin will not
surface the second one at all. Users on a media server should either separate
sub variants into their own files (`{language}` in the naming template) or use
`--fetch-subs`, which yields a real subtitle track instead.

## Design notes

* The interface is a single pluggable call:
  `detect_offset(dub_path, ref_path) -> AlignResult(offset, confidence, drift,
  tempo_ratio, …)` — the correlation implementation can be swapped without
  touching the pipeline.
* Only **numpy** is required (`rfft`/`irfft`); no scipy, no fingerprinting
  libraries.
* End-to-end analysis cost is ~2 s per episode: everything after decoding
  operates on 100 Hz envelopes, not raw audio.
* The technique is the same family as [ffsubsync](https://github.com/smacke/ffsubsync)
  uses for subtitle alignment, adapted to audio-vs-audio with the tempo scan
  added for speed-changed dubs.
* Unit tests ([`tests/test_dubsync_align.py`](../tests/test_dubsync_align.py))
  synthesise impulse-bed signals with known shifts, PAL-speed drift, and
  uncorrelated noise, asserting recovered offsets/tempi and that noise stays
  below the confidence threshold.
  [`tests/test_multilang_merge.py`](../tests/test_multilang_merge.py) covers
  the download-path reuse end to end: a deliberately delayed second language
  must come back aligned, tagged and non-default, in a single file.
