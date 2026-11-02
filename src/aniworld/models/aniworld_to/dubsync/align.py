"""Automatic audio alignment between a dub track and a target's own audio.

The dub and the target carry *different dialogue* (German vs JP/EN), so speech
cannot be correlated directly. What both share is the music/SFX bed: loud
musical hits and effects dominate an onset-energy envelope regardless of the
spoken language. Cross-correlating those envelopes (via FFT) locks onto the
shared bed and yields the constant offset between the two tracks; the peak's
sharpness (a z-score against the rest of the correlation) is the confidence.

Linear drift (e.g. PAL-sourced dubs running ~4% fast) smears the correlation
peak, so a hierarchical tempo scan time-stretches the dub envelope over a grid
of speed factors (couple of percent around 1.0, refined in stages down to
1e-5) and keeps the factor whose correlation peak is sharpest -- the correct
tempo un-smears the peak dramatically, and a wrong one never beats the
identity baseline by the required margin. A detected factor can be corrected
with :func:`resample_dub` via ``atempo`` -- which re-encodes the dub audio, so
the pipeline only takes that path when the user opts in.

Only :mod:`numpy` is needed (FFT correlation), no scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import ffmpeg
import numpy as np

from ....config import logger
from ...common.common import _run_ffmpeg_with_progress

# Decode/analysis parameters. 8 kHz mono is plenty for an energy envelope; the
# envelope itself is sampled at 100 Hz (10 ms resolution, refined below by
# parabolic peak interpolation).
_DECODE_RATE = 8000
_ENVELOPE_RATE = 100

#: Largest constant offset (seconds) the global search considers.
DEFAULT_MAX_OFFSET = 600.0

#: Peak z-score below which a detected offset should not be trusted. A true
#: match gives a needle-sharp peak (z >> 10); uncorrelated tracks stay ~3-5.
DEFAULT_MIN_CONFIDENCE = 8.0

#: Offset change across the whole episode (seconds) beyond which drift is
#: reported (and correctable via resample).
DRIFT_THRESHOLD = 0.2

# Tempo scan: stretch factors 1.0 +/- _TEMPO_SPAN are searched at the first
# step size, then the best factor is refined at each finer step. The final
# step bounds the tempo error at 1e-5 (~14 ms residual drift over 24 min).
_TEMPO_SPAN = 0.06  # covers PAL <-> film (~4.3%) with headroom
_TEMPO_STEPS = (2e-3, 2e-4, 2e-5)

# Speed ratios real dubs actually differ by: PAL speed-up of film content
# (25/23.976, 25/24) and the NTSC 1000/1001 slip, in both directions. The
# correlation peak is needle-thin in tempo -- over a 50-minute episode a
# factor 7e-4 off already smears it by two seconds -- so a coarse grid steps
# right over the true ratio and finds nothing. These are probed (with a small
# neighbourhood) *before* the grid, which only runs when none of them fits.
_STANDARD_RATIOS = tuple(
    r
    for base in (25 / 23.976, 25 / 24, 24 / 23.976)
    for r in (base, 1 / base)
)
_STANDARD_SPAN = 1e-4
_STANDARD_STEP = 2e-5

# The generic grid's first step must keep the stretch error well under a
# second at the episode's end, or the peak at the true factor is never seen.
_GRID_MAX_SMEAR = 0.3  # seconds

# Piecewise ("cut") detection: TV versions drop a few seconds at act breaks,
# so one constant offset only fits part of the episode. The dub is split into
# chunks that are aligned individually within +/- _CUT_SEARCH of the global
# offset; runs of chunks agreeing on an offset become segments.
_CUT_CHUNK = 60.0  # seconds per chunk
_CUT_SEARCH = 90.0  # max offset change (s) considered relative to global
_CUT_TOLERANCE = 0.12  # chunks within this many seconds share a segment
_CUT_MIN_SHIFT = 0.15  # smaller jumps between segments are ignored
# Placing a cut between two runs: short sub-windows across the transition are
# aligned individually and labelled "old shift" / "new shift" when their peak
# is sharp (periodic music matches both shifts but only with a blunt peak, so
# it stays unlabelled instead of dragging the cut). The split that agrees
# with the most labels bounds the cut to an interval; inside it the cut goes
# where the patched span is quietest: where the dub is missing material, the
# stretch of the *reference* that will play against padded silence (the fade
# the TV cut removed); where the dub has extra, the stretch of the dub that
# is dropped. Without usable labels, the summed-correlation score (old shift
# before, new shift after) and its noise plateau stand in for the interval.
_CUT_SUBWIN = 20.0  # seconds per labelling sub-window
_CUT_SUBHOP = 2.0
_CUT_SUBSEARCH = 15.0  # lag search (s) beyond the two candidate shifts
_CUT_SUB_CONF = 6.0
_CUT_PLATEAU = 10.0
_CUT_SNAP = 8.0
_CUT_MIN_SPAN = 0.2  # seconds; shortest span averaged when judging quietness

# Shortest dub (seconds) for which drift analysis is attempted at all.
_DRIFT_MIN_DURATION = 300.0

# A non-identity tempo must beat the identity peak by this much z-score (and
# clear DEFAULT_MIN_CONFIDENCE) before drift is claimed -- otherwise noise
# could pick a wrong speed. A genuinely drifted dub un-smears dramatically.
_TEMPO_MARGIN = 5.0

# Seconds around the peak excluded when estimating the correlation noise floor.
_PEAK_EXCLUDE = 1.0

# Japanese language tags used to pick the preferred reference audio stream.
_JP_TAGS = {"jpn", "ja", "jp", "japanese"}


@dataclass
class AlignResult:
    """Outcome of aligning one dub against one reference track.

    ``offset`` is the best *constant* delay (seconds, may be negative) to apply
    to the dub via ``-itsoffset``; ``confidence`` is the correlation peak
    z-score. When drift was detected, ``drift`` is how much the offset changes
    across the whole episode, ``tempo_ratio`` the ``atempo`` factor that
    corrects it, and ``post_tempo_offset`` the residual constant offset to use
    *after* the dub has been retimed with that factor.
    """

    offset: float
    confidence: float
    drift: float = 0.0
    drift_confidence: float = 0.0
    tempo_ratio: float = 1.0
    post_tempo_offset: float = 0.0
    #: Piecewise map (only with ``find_cuts=True``): one :class:`Segment` per
    #: stretch of the dub sharing a constant offset, in *post-tempo* dub time.
    #: A single segment means no cuts were found.
    segments: List["Segment"] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return abs(self.drift) > DRIFT_THRESHOLD

    @property
    def has_cuts(self) -> bool:
        return len(self.segments) > 1


@dataclass
class Segment:
    """A stretch of the (post-tempo) dub that lines up with the reference at
    a constant ``shift`` (seconds): dub time ``t`` plays at reference time
    ``t + shift``. ``end`` is ``None`` for the last segment (runs to EOF)."""

    start: float
    end: Optional[float]
    shift: float


def _pick_reference_stream(path: Path) -> int:
    """Audio-stream index (``0:a:N``) to correlate against: JP if tagged, else
    the first audio stream. Raises if the file has no audio at all."""

    try:
        probe = ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        raise RuntimeError(f"Could not probe reference audio: {path}") from exc

    audio_streams = [
        s for s in probe.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise RuntimeError(f"No audio stream in reference file: {path}")

    for idx, stream in enumerate(audio_streams):
        lang = (stream.get("tags", {}).get("language") or "").lower()
        if lang in _JP_TAGS:
            return idx
    return 0


def _decode_mono(path: Path, stream_index: int = 0) -> np.ndarray:
    """Decode one audio stream of *path* to mono float32 at ``_DECODE_RATE``."""

    try:
        out, _ = (
            ffmpeg.input(str(path))
            .output(
                "pipe:",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar=_DECODE_RATE,
                map=f"0:a:{stream_index}",
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        tail = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg failed decoding {Path(path).name}: "
            + (tail[-1] if tail else "unknown error")
        ) from exc

    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _onset_envelope(samples: np.ndarray) -> np.ndarray:
    """Half-wave-rectified log-energy onset envelope at ``_ENVELOPE_RATE`` Hz,
    z-normalised. Loud transient events (music hits, SFX) dominate it, which is
    exactly the language-independent signal shared between dub and original."""

    hop = _DECODE_RATE // _ENVELOPE_RATE
    n_frames = len(samples) // hop
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    env = np.log1p(rms * 100.0)

    onset = np.diff(env, prepend=env[:1])
    np.maximum(onset, 0.0, out=onset)

    std = onset.std()
    if std < 1e-8:
        return np.zeros_like(onset)
    return (onset - onset.mean()) / std


def _xcorr_peak_fft(
    ref_fft: np.ndarray,
    n: int,
    qry_env: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> tuple[float, float]:
    """Correlation-peak search against a precomputed reference spectrum, so a
    tempo scan pays for the reference FFT only once."""

    if len(qry_env) == 0:
        return 0.0, 0.0

    corr = np.fft.irfft(ref_fft * np.conj(np.fft.rfft(qry_env, n)), n)

    # corr[k] = sum(ref[m + k] * qry[m]); negative lags wrap to the array end.
    lags = np.arange(n)
    lags[lags > n // 2] -= n

    valid = (lags >= min_lag) & (lags <= max_lag)
    if not valid.any():
        return 0.0, 0.0

    valid_idx = np.flatnonzero(valid)
    seg = corr[valid_idx]
    best = int(valid_idx[np.argmax(seg)])
    peak_lag = int(lags[best])
    peak_val = corr[best]

    # Confidence: peak z-score against the searched range, excluding the
    # immediate neighbourhood of the peak itself.
    exclude = int(_PEAK_EXCLUDE * _ENVELOPE_RATE)
    noise = seg[np.abs(lags[valid_idx] - peak_lag) > exclude]
    if len(noise) < 8:
        return float(peak_lag), 0.0
    noise_std = noise.std()
    confidence = (
        float((peak_val - noise.mean()) / noise_std) if noise_std > 1e-12 else 0.0
    )

    # Parabolic interpolation for sub-envelope-sample lag precision.
    refined = float(peak_lag)
    prev_i, next_i = (best - 1) % n, (best + 1) % n
    y0, y1, y2 = corr[prev_i], peak_val, corr[next_i]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) > 1e-12:
        delta = 0.5 * (y0 - y2) / denom
        if abs(delta) <= 1.0:
            refined += float(delta)

    return refined, confidence


def _xcorr_peak(
    ref_env: np.ndarray,
    qry_env: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> tuple[float, float]:
    """FFT cross-correlation peak of *qry_env* against *ref_env*.

    Returns ``(lag, confidence)`` where ``lag`` (envelope samples, may be
    fractional after parabolic refinement) is how far *qry* must be delayed to
    line up with *ref*, and ``confidence`` is the peak's z-score against the
    rest of the searched correlation range.
    """

    if len(ref_env) == 0 or len(qry_env) == 0:
        return 0.0, 0.0

    n = 1 << int(len(ref_env) + len(qry_env) - 1).bit_length()
    return _xcorr_peak_fft(np.fft.rfft(ref_env, n), n, qry_env, min_lag, max_lag)


def _stretch(env: np.ndarray, factor: float) -> np.ndarray:
    """Time-stretch an envelope by *factor* (linear interpolation). A dub that
    plays ``factor`` times too fast lines up with the reference after its
    envelope is stretched by ``factor``."""

    if factor == 1.0 or len(env) == 0:
        return env
    idx = np.arange(int(len(env) * factor)) / factor
    return np.interp(idx, np.arange(len(env)), env).astype(env.dtype)


def _energy_envelope(samples: np.ndarray) -> np.ndarray:
    """Log RMS energy per envelope frame -- used to place cut boundaries in
    the quietest spot (act breaks fade to silence)."""

    hop = _DECODE_RATE // _ENVELOPE_RATE
    n_frames = len(samples) // hop
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return np.log1p(rms * 100.0).astype(np.float32)


def _tempo_scan(
    ref_env: np.ndarray,
    dub_env: np.ndarray,
    max_lag: int,
) -> tuple[float, float, float]:
    """Find the tempo factor whose stretched-dub correlation peak is sharpest.

    1. The standard broadcast ratios (:data:`_STANDARD_RATIOS`) are probed
       first, each with a small neighbourhood. If one of them clearly beats
       the identity peak it is polished and returned.
    2. Otherwise a coarse-to-fine grid over ``1 +/- _TEMPO_SPAN``. When the
       identity peak is weak (the drifted case) the first step is made fine
       enough for this duration that the true factor cannot be stepped over.

    Returns ``(factor, lag, confidence)``.
    """

    n = 1 << int(len(ref_env) + len(dub_env) * (1 + _TEMPO_SPAN)).bit_length()
    ref_fft = np.fft.rfft(ref_env, n)
    cache: dict = {}

    def probe(factor: float) -> tuple[float, float]:
        key = round(float(factor), 8)
        if key not in cache:
            cache[key] = _xcorr_peak_fft(
                ref_fft, n, _stretch(dub_env, key), -max_lag, max_lag
            )
        return cache[key]

    def best_of(factors) -> tuple[float, float, float]:
        best = (1.0, 0.0, -np.inf)
        for factor in factors:
            lag, conf = probe(factor)
            if conf > best[2]:
                best = (round(float(factor), 8), lag, conf)
        return best

    def refine(center: float, span: float, steps: Sequence[float]):
        best = (center, *probe(center))
        for step in steps:
            cand = best_of(
                np.arange(best[0] - span, best[0] + span + step / 2, step)
            )
            if cand[2] > best[2]:
                best = cand
            span = step
        return best

    _, identity_conf = probe(1.0)

    neighbourhood = np.arange(
        -_STANDARD_SPAN, _STANDARD_SPAN + _STANDARD_STEP / 2, _STANDARD_STEP
    )
    std = best_of(r + d for r in _STANDARD_RATIOS for d in neighbourhood)
    if std[2] >= DEFAULT_MIN_CONFIDENCE and std[2] >= identity_conf + _TEMPO_MARGIN:
        return refine(std[0], _STANDARD_STEP, (4e-6, 1e-6))

    steps = _TEMPO_STEPS
    if identity_conf < DEFAULT_MIN_CONFIDENCE:
        duration = max(len(dub_env) / _ENVELOPE_RATE, 1.0)
        first = min(_TEMPO_STEPS[0], _GRID_MAX_SMEAR / duration)
        steps = (first,) + tuple(s for s in _TEMPO_STEPS if s < first / 2)

    best = best_of(np.arange(1.0 - _TEMPO_SPAN, 1.0 + _TEMPO_SPAN + steps[0] / 2, steps[0]))
    if len(steps) > 1:
        best = max(best, refine(best[0], steps[0], steps[1:]), key=lambda b: b[2])
    return best


def _label_interval(
    ref_env: np.ndarray, dub_env: np.ndarray, lo: int, hi: int, sa: float, sb: float
) -> Optional[tuple[int, int]]:
    """Bound the cut between shifts *sa* -> *sb* inside dub ``[lo, hi)`` by
    labelling sharp-peaked sub-windows; ``None`` when there aren't labels on
    both sides."""

    rate = _ENVELOPE_RATE
    win, hop = int(_CUT_SUBWIN * rate), int(_CUT_SUBHOP * rate)
    pad = int(_CUT_SUBSEARCH * rate)
    tol = _CUT_TOLERANCE * rate
    labels = []  # (centre, is_new_shift)
    for s0 in range(lo, hi - win + 1, hop):
        q = dub_env[s0 : s0 + win]
        r_lo = max(s0 + int(min(sa, sb)) - pad, 0)
        r_hi = min(s0 + win + int(np.ceil(max(sa, sb))) + pad, len(ref_env))
        if r_hi - r_lo <= win:
            continue
        pos, conf = _xcorr_peak(ref_env[r_lo:r_hi], q, 0, r_hi - r_lo - win)
        if conf < _CUT_SUB_CONF:
            continue
        shift = r_lo + pos - s0
        if abs(shift - sa) <= tol:
            labels.append((s0 + win // 2, False))
        elif abs(shift - sb) <= tol:
            labels.append((s0 + win // 2, True))

    if not any(new for _, new in labels) or all(new for _, new in labels):
        return None

    # split k puts labels[:k] before the cut; score = labels it agrees with
    n = len(labels)
    old_before = np.concatenate([[0], np.cumsum([not new for _, new in labels])])
    new_after = np.concatenate([np.cumsum([new for _, new in labels][::-1])[::-1], [0]])
    agree = old_before + new_after
    ties = np.flatnonzero(agree == agree.max())
    left = lambda k: labels[k - 1][0] if k > 0 else lo  # noqa: E731
    right = lambda k: labels[k][0] if k < n else hi  # noqa: E731
    # A label follows where a window's onsets are, not strictly where most
    # of its duration is, so the bound is softened by a quarter window.
    slack = win // 4
    return (
        max(min(left(k) for k in ties) - slack, lo),
        min(max(right(k) for k in ties) + slack, hi),
    )


def _place_cut(
    ref_env: np.ndarray,
    dub_env: np.ndarray,
    lo: int,
    hi: int,
    sa: float,
    sb: float,
    dub_energy: Optional[np.ndarray],
    ref_energy: Optional[np.ndarray],
) -> int:
    """Dub sample index of the cut between shifts *sa* and *sb* (see the
    ``_CUT_*`` notes above)."""

    rate = _ENVELOPE_RATE
    idx = np.arange(lo, hi)

    def products(shift: float) -> np.ndarray:
        ref_idx = idx + int(round(shift))
        ok = (ref_idx >= 0) & (ref_idx < len(ref_env))
        out = np.zeros(len(idx))
        out[ok] = dub_env[idx[ok]] * ref_env[ref_idx[ok]]
        return out

    pa, pb = products(sa), products(sb)
    score = (np.concatenate([[0.0], np.cumsum(pa)]) + (
        pb.sum() - np.concatenate([[0.0], np.cumsum(pb)])
    ))[: len(idx)]

    interval = _label_interval(ref_env, dub_env, lo, hi, sa, sb)
    if interval is not None:
        cand = np.arange(max(interval[0], lo), min(interval[1], hi) + 1) - lo
        cand = cand[cand < len(idx)]
    else:
        best = int(np.argmax(score))
        margin = float(np.std(pa - pb)) * np.sqrt(_CUT_PLATEAU * rate)
        cand = np.flatnonzero(score >= score[best] - margin)
        cand = cand[np.abs(cand - best) <= int(_CUT_SNAP * rate)]
    if not len(cand):
        return lo + int(np.argmax(score))

    gap = int(round(sb - sa))
    if gap > 0:
        # padded silence plays against reference [cut + sa, cut + sb)
        energy, offset_into = ref_energy, int(round(sa))
    else:
        # dub [cut, cut + |gap|) is dropped
        energy, offset_into = dub_energy, 0
    span = max(abs(gap), int(_CUT_MIN_SPAN * rate), 1)
    if energy is not None and len(energy) > span:
        csum = np.concatenate([[0.0], np.cumsum(energy, dtype=np.float64)])
        starts = lo + cand + offset_into
        ok = (starts >= 0) & (starts + span <= len(energy))
        if ok.any():
            c, st = cand[ok], starts[ok]
            cost = (csum[st + span] - csum[st]) / span
            return lo + int(c[np.argmin(cost)])
    return lo + int(cand[np.argmax(score[cand])])


def _find_segments(
    ref_env: np.ndarray,
    dub_env: np.ndarray,
    anchor_lag: float,
    dub_energy: Optional[np.ndarray] = None,
    ref_energy: Optional[np.ndarray] = None,
) -> List[Segment]:
    """Split the dub into stretches that each line up at a constant shift.

    *dub_env* must already be in reference timescale (i.e. stretched by the
    detected tempo factor); *anchor_lag* is the global peak lag (envelope
    samples) and bounds the per-chunk search. Each :data:`_CUT_CHUNK` chunk
    is aligned on its own; consecutive confident chunks that agree form a
    run; the boundary between two runs is placed by :func:`_place_cut`.
    *dub_energy* / *ref_energy* are :func:`_energy_envelope` arrays (the dub
    one stretched like *dub_env*).
    """

    rate = _ENVELOPE_RATE
    ch = int(_CUT_CHUNK * rate)
    search = int(_CUT_SEARCH * rate)
    anchor = int(round(anchor_lag))
    tol = _CUT_TOLERANCE * rate

    chunks = []  # (start, shift_samples, confidence)
    for start in range(0, max(len(dub_env) - ch // 2, 1), ch):
        q = dub_env[start : start + ch]
        lo = max(start + anchor - search, 0)
        hi = min(start + len(q) + anchor + search, len(ref_env))
        if hi - lo < len(q) + 2 * rate:
            continue
        pos, conf = _xcorr_peak(ref_env[lo:hi], q, 0, hi - lo - len(q))
        chunks.append((start, lo + pos - start, conf))

    runs: List[dict] = []
    for start, shift, conf in chunks:
        if conf < DEFAULT_MIN_CONFIDENCE:
            continue
        if runs and abs(shift - float(np.median(runs[-1]["shifts"]))) <= tol:
            runs[-1]["shifts"].append(shift)
            runs[-1]["last"] = start
            runs[-1]["peak"] = max(runs[-1]["peak"], conf)
        else:
            runs.append({"first": start, "last": start, "shifts": [shift], "peak": conf})

    # A lone chunk that disagrees with its neighbours is far more likely a
    # spurious peak than a sub-minute act; real acts span several chunks.
    runs = [r for r in runs if len(r["shifts"]) > 1 or r["peak"] >= 12.0] or runs

    merged: List[dict] = []
    for run in runs:
        if merged and abs(
            float(np.median(run["shifts"])) - float(np.median(merged[-1]["shifts"]))
        ) < _CUT_MIN_SHIFT * rate:
            merged[-1]["shifts"].extend(run["shifts"])
            merged[-1]["last"] = run["last"]
        else:
            merged.append(dict(run, shifts=list(run["shifts"])))

    if not merged:
        return [Segment(0.0, None, anchor_lag / rate)]

    shifts = [float(np.median(r["shifts"])) for r in merged]
    bounds: List[float] = []
    for a, b, sa, sb in zip(merged, merged[1:], shifts, shifts[1:]):
        lo = a["last"]
        hi = min(b["first"] + ch, len(dub_env))
        cut = _place_cut(ref_env, dub_env, lo, hi, sa, sb, dub_energy, ref_energy)
        bounds.append(cut / rate)

    segments = []
    starts = [0.0] + bounds
    ends: List[Optional[float]] = bounds + [None]
    for start, end, shift in zip(starts, ends, shifts):
        segments.append(Segment(start, end, shift / rate))
    return segments


def detect_offset(
    dub_path,
    ref_path,
    max_offset: float = DEFAULT_MAX_OFFSET,
    check_drift: bool = True,
    find_cuts: bool = False,
) -> AlignResult:
    """Detect the offset between *dub_path* and *ref_path*'s audio.

    *ref_path* may be a full video container; its JP audio stream is preferred
    as the reference (falling back to the first audio stream). The returned
    :class:`AlignResult` always carries the best constant offset; drift fields
    are populated when the tracks are long enough for the two-window check.
    With *find_cuts*, :attr:`AlignResult.segments` holds the piecewise map
    (post-tempo dub time) -- see :func:`render_dub` to apply it.
    """

    dub_path = Path(dub_path)
    ref_path = Path(ref_path)

    ref_stream = _pick_reference_stream(ref_path)
    ref_samples = _decode_mono(ref_path, ref_stream)
    ref_env = _onset_envelope(ref_samples)
    dub_samples = _decode_mono(dub_path)
    dub_env = _onset_envelope(dub_samples)

    rate = _ENVELOPE_RATE
    max_lag = int(max_offset * rate)
    lag, confidence = _xcorr_peak(ref_env, dub_env, -max_lag, max_lag)
    offset = lag / rate

    result = AlignResult(
        offset=offset, confidence=confidence, post_tempo_offset=offset
    )
    logger.debug(
        f"[DUBSYNC] global offset {offset:+.3f}s (confidence z={confidence:.1f}) "
        f"for {dub_path.name}"
    )

    factor, anchor = 1.0, lag
    dub_duration = len(dub_env) / rate
    if check_drift and dub_duration >= _DRIFT_MIN_DURATION:
        # Linear drift smears the identity-tempo peak; scanning stretch
        # factors un-smears it at the true speed ratio. A non-identity factor
        # must beat the identity peak by a clear margin so noise can't pick
        # a wrong speed.
        t_factor, t_lag, t_conf = _tempo_scan(ref_env, dub_env, max_lag)
        logger.debug(
            f"[DUBSYNC] tempo scan: x{t_factor:.6f} z={t_conf:.1f} "
            f"(identity z={confidence:.1f})"
        )
        if not (
            t_factor == 1.0
            or t_conf < DEFAULT_MIN_CONFIDENCE
            or t_conf < confidence + _TEMPO_MARGIN
        ):
            result.confidence = t_conf
            result.drift = (t_factor - 1.0) * dub_duration
            result.drift_confidence = t_conf
            if result.has_drift:
                # Slowing the dub by the detected factor (atempo = 1/factor)
                # removes the drift; the scan's peak lag is the remaining
                # constant offset.
                result.tempo_ratio = 1.0 / t_factor
                result.post_tempo_offset = t_lag / rate
                factor, anchor = t_factor, t_lag

    if find_cuts and result.confidence >= DEFAULT_MIN_CONFIDENCE:
        energy = _energy_envelope(dub_samples)
        result.segments = _find_segments(
            ref_env,
            _stretch(dub_env, factor),
            anchor,
            _stretch(energy, factor) if factor != 1.0 else energy,
            _energy_envelope(ref_samples),
        )
        if result.has_cuts:
            logger.debug(
                "[DUBSYNC] cuts: "
                + ", ".join(
                    f"{s.start:.1f}s->{s.shift:+.2f}s" for s in result.segments
                )
            )

    return result


def render_dub(
    dub_path,
    dest_path,
    tempo_ratio: float = 1.0,
    segments: Optional[Sequence[Segment]] = None,
    label: str = "",
    acodec: str = "aac",
) -> Path:
    """Retime *dub_path* and/or re-assemble it around cuts in one pass.

    The output starts at the first segment's position, so the caller still
    applies ``segments[0].shift`` as the constant ``-itsoffset``. Every later
    segment is moved by its shift relative to the first: silence is padded in
    where the dub is missing material (the usual act-break cut), and dub
    material is dropped where it has extra.

    Each segment is its own seeked input -- splitting one decoded stream
    would buffer the whole episode in memory before concat reaches the last
    part. Re-encodes (AAC 192k by default: the dub is already a lossy stream
    rip, a lossless copy would only balloon the file).
    """

    dub_path = Path(dub_path)
    dest_path = Path(dest_path)
    segs = list(segments or [Segment(0.0, None, 0.0)])
    retime = abs(tempo_ratio - 1.0) > 1e-9
    # Segment times are post-tempo; input seeks happen before atempo.
    pre = (lambda t: t * tempo_ratio) if retime else (lambda t: t)

    parts = []
    trim_head = 0.0
    for i, seg in enumerate(segs):
        start = seg.start + trim_head
        trim_head = 0.0
        if seg.end is not None and start >= seg.end:
            trim_head = start - seg.end
            continue
        in_kwargs = {}
        if start > 0:
            in_kwargs["ss"] = f"{pre(start):.4f}"
        if seg.end is not None:
            in_kwargs["t"] = f"{pre(seg.end - start):.4f}"
        part = ffmpeg.input(str(dub_path), **in_kwargs).audio
        if retime:
            part = part.filter("atempo", f"{tempo_ratio:.8f}")
        if i + 1 < len(segs):
            gap = segs[i + 1].shift - seg.shift
            if gap > 0:
                part = part.filter("apad", pad_dur=f"{gap:.4f}")
            elif gap < 0:
                trim_head = -gap
        parts.append(part)

    stream = parts[0] if len(parts) == 1 else ffmpeg.concat(*parts, v=0, a=1)

    encode_kwargs = {"acodec": acodec}
    if acodec == "aac":
        encode_kwargs["audio_bitrate"] = "192k"

    logger.debug(
        f"[DUBSYNC] rendering dub (atempo={tempo_ratio:.6f}, "
        f"{len(parts)} segment(s)) -> {dest_path.name}"
    )
    _run_ffmpeg_with_progress(
        ffmpeg.output(stream, str(dest_path), **encode_kwargs), label=label
    )
    return dest_path


def resample_dub(
    dub_path, dest_path, tempo_ratio: float, label: str = "", acodec: str = "flac"
) -> Path:
    """Retime *dub_path* by *tempo_ratio* (``atempo``) to correct linear drift.

    This is the one lossy step in DubSync: ``atempo`` requires decoding, so the
    result is re-encoded. FLAC by default, so no further generational loss
    after the decode; callers retiming an already-lossy stream rip (where a
    lossless track would balloon the file) can ask for a lossy *acodec*
    instead. Only called when the user opted in to resampling.
    """

    dub_path = Path(dub_path)
    dest_path = Path(dest_path)

    logger.debug(
        f"[DUBSYNC] retiming dub with atempo={tempo_ratio:.6f} -> {dest_path.name}"
    )

    encode_kwargs = {"acodec": acodec}
    if acodec == "aac":
        encode_kwargs["audio_bitrate"] = "192k"

    node = ffmpeg.input(str(dub_path)).output(
        str(dest_path),
        af=f"atempo={tempo_ratio:.8f}",
        map="0:a:0",
        **encode_kwargs,
    )
    _run_ffmpeg_with_progress(node, label=label)
    return dest_path
