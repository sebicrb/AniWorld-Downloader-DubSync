"""Unit tests for the DubSync alignment engine (synthetic signals).

The correlation math is tested directly on synthesized envelopes/PCM: two
tracks sharing an impulse "music bed" but carrying different "dialogue" noise
must align at the known shift with high confidence, while unrelated noise must
come back low-confidence. The full decode path is exercised through real WAV
files when ffmpeg is available.
"""

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import subprocess

from aniworld.models.aniworld_to.dubsync.align import (
    _DECODE_RATE,
    _ENVELOPE_RATE,
    DEFAULT_MIN_CONFIDENCE,
    Segment,
    _find_segments,
    _onset_envelope,
    _stretch,
    _tempo_scan,
    _xcorr_peak,
    detect_offset,
    render_dub,
)

def _impulse_bed(rng, duration_s: float, n_events: int = 60) -> np.ndarray:
    """Silent track with short loud noise bursts at random times -- a stand-in
    for the shared music/SFX bed."""

    samples = np.zeros(int(duration_s * _DECODE_RATE), dtype=np.float32)
    burst_len = _DECODE_RATE // 10  # 100 ms
    positions = rng.integers(0, len(samples) - burst_len, n_events)
    for pos in positions:
        samples[pos : pos + burst_len] += 0.7 * rng.standard_normal(burst_len).astype(
            np.float32
        )
    return samples


def _dialogue(rng, duration_s: float, level: float = 0.08) -> np.ndarray:
    """Continuous low-level noise -- a stand-in for (language-specific) speech."""

    return level * rng.standard_normal(int(duration_s * _DECODE_RATE)).astype(
        np.float32
    )


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_DECODE_RATE)
        wav.writeframes(pcm.tobytes())


def test_xcorr_recovers_known_shift():
    rng = np.random.default_rng(42)
    bed = _impulse_bed(rng, 120.0)
    ref_env = _onset_envelope(bed + _dialogue(rng, 120.0))

    shift_s = 2.5
    pad = np.zeros(int(shift_s * _DECODE_RATE), dtype=np.float32)
    dub_env = _onset_envelope(np.concatenate([pad, bed]) + _dialogue(rng, 122.5))

    max_lag = 30 * _ENVELOPE_RATE
    lag, confidence = _xcorr_peak(ref_env, dub_env, -max_lag, max_lag)

    # Dub content occurs later, so it must be advanced: negative lag.
    assert lag / _ENVELOPE_RATE == pytest.approx(-shift_s, abs=0.05)
    assert confidence >= DEFAULT_MIN_CONFIDENCE


def test_uncorrelated_noise_is_low_confidence():
    rng = np.random.default_rng(43)
    env_a = _onset_envelope(_dialogue(rng, 120.0, level=0.3))
    env_b = _onset_envelope(_dialogue(rng, 120.0, level=0.3))

    max_lag = 30 * _ENVELOPE_RATE
    _, confidence = _xcorr_peak(env_a, env_b, -max_lag, max_lag)

    assert confidence < DEFAULT_MIN_CONFIDENCE


def test_silence_yields_zero_confidence():
    silent = _onset_envelope(np.zeros(60 * _DECODE_RATE, dtype=np.float32))
    lag, confidence = _xcorr_peak(silent, silent, -100, 100)
    assert lag == 0.0
    assert confidence == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_detect_offset_end_to_end(tmp_path):
    rng = np.random.default_rng(44)
    bed = _impulse_bed(rng, 90.0)
    ref = bed + _dialogue(rng, 90.0)

    shift_s = 1.5
    pad = np.zeros(int(shift_s * _DECODE_RATE), dtype=np.float32)
    dub = np.concatenate([pad, bed]) + _dialogue(rng, 91.5)

    ref_path = tmp_path / "ref.wav"
    dub_path = tmp_path / "dub.wav"
    _write_wav(ref_path, ref)
    _write_wav(dub_path, dub)

    result = detect_offset(dub_path, ref_path, max_offset=30.0)

    assert result.offset == pytest.approx(-shift_s, abs=0.05)
    assert result.confidence >= DEFAULT_MIN_CONFIDENCE
    # 90 s is far below the drift-check minimum, so no drift is reported.
    assert result.drift == 0.0
    assert result.tempo_ratio == 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_detect_offset_finds_pal_drift(tmp_path):
    rng = np.random.default_rng(45)
    speed = 25 / 23.976  # PAL speed-up: the classic German-dub drift case
    bed = _impulse_bed(rng, 600.0, n_events=300)
    ref = bed + _dialogue(rng, 600.0)

    # Dub plays `speed` times too fast: dub[i] = bed[i * speed].
    idx = np.arange(int(len(bed) / speed)) * speed
    dub = np.interp(idx, np.arange(len(bed)), bed).astype(np.float32)
    dub += _dialogue(rng, len(dub) / _DECODE_RATE)

    ref_path = tmp_path / "ref.wav"
    dub_path = tmp_path / "dub.wav"
    _write_wav(ref_path, ref)
    _write_wav(dub_path, dub)

    result = detect_offset(dub_path, ref_path, max_offset=60.0)

    assert result.has_drift
    assert result.tempo_ratio == pytest.approx(1 / speed, abs=1e-4)
    assert result.drift_confidence >= DEFAULT_MIN_CONFIDENCE
    assert result.post_tempo_offset == pytest.approx(0.0, abs=0.1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_no_false_drift_on_clean_alignment(tmp_path):
    rng = np.random.default_rng(46)
    bed = _impulse_bed(rng, 600.0, n_events=300)
    ref = bed + _dialogue(rng, 600.0)
    dub = bed + _dialogue(rng, 600.0)

    ref_path = tmp_path / "ref.wav"
    dub_path = tmp_path / "dub.wav"
    _write_wav(ref_path, ref)
    _write_wav(dub_path, dub)

    result = detect_offset(dub_path, ref_path, max_offset=60.0)

    assert result.offset == pytest.approx(0.0, abs=0.05)
    assert not result.has_drift
    assert result.tempo_ratio == 1.0


# ----- long-episode tempo + act-break cuts (envelope level) -----


def _sparse_env(rng, duration_s: float, rate_hz: float = 0.5) -> np.ndarray:
    """Envelope with random onset spikes (music/SFX hits) at ~rate_hz."""

    n = int(duration_s * _ENVELOPE_RATE)
    env = np.zeros(n, dtype=np.float32)
    hits = rng.integers(0, n, int(duration_s * rate_hz))
    env[hits] = rng.uniform(2.0, 6.0, len(hits))
    return env


def _noisy(rng, env: np.ndarray, level: float = 0.6) -> np.ndarray:
    return (env + level * rng.standard_normal(len(env))).astype(np.float32)


def test_tempo_scan_finds_pal_ratio_on_long_episode():
    """50 minutes at PAL speed: a 2e-3 grid steps over the needle-thin peak
    (the Animal Kingdom failure); the standard-ratio probe must land on it."""

    rng = np.random.default_rng(46)
    speed = 25 / 23.976
    ref = _sparse_env(rng, 3000.0)
    dub = _stretch(ref, 1 / speed)  # dub plays `speed` times too fast

    factor, lag, conf = _tempo_scan(_noisy(rng, ref), _noisy(rng, dub), 60 * _ENVELOPE_RATE)

    assert factor == pytest.approx(speed, abs=2e-5)
    assert conf >= DEFAULT_MIN_CONFIDENCE
    assert lag / _ENVELOPE_RATE == pytest.approx(0.0, abs=0.1)


def test_find_segments_recovers_act_break_cuts():
    """German TV cuts drop a few seconds at each act break: the dub must be
    split there, each part carrying its own shift."""

    rng = np.random.default_rng(47)
    ref = _sparse_env(rng, 1800.0, rate_hz=1.0)
    r = _ENVELOPE_RATE
    # Remove 2.5 s at ref 400 s and 3.0 s at ref 1100 s from the dub.
    dub = np.concatenate(
        [ref[: 400 * r], ref[int(402.5 * r) : 1100 * r], ref[1103 * r :]]
    )

    # The removed seconds are the fade to black: silent in the reference.
    ref_energy = (3.0 + 0.3 * rng.standard_normal(len(ref))).astype(np.float32)
    ref_energy[400 * r : int(402.5 * r)] = 0.1
    ref_energy[1100 * r : 1103 * r] = 0.1

    segs = _find_segments(
        _noisy(rng, ref), _noisy(rng, dub), 0.0, ref_energy=ref_energy
    )

    assert len(segs) == 3
    assert [s.shift for s in segs] == pytest.approx([0.0, 2.5, 5.5], abs=0.05)
    # Correlation only bounds the cut to a few seconds (no hits inside a
    # fade); the reference's silent fade pins it.
    assert segs[1].start == pytest.approx(400.0, abs=0.3)
    assert segs[2].start == pytest.approx(1097.5, abs=0.3)
    assert segs[-1].end is None


def test_find_segments_single_segment_without_cuts():
    rng = np.random.default_rng(48)
    ref = _sparse_env(rng, 900.0)
    pad = np.zeros(int(1.2 * _ENVELOPE_RATE), dtype=np.float32)
    dub = np.concatenate([ref[len(pad):], pad])  # dub runs 1.2 s early

    segs = _find_segments(_noisy(rng, ref), _noisy(rng, dub), 1.2 * _ENVELOPE_RATE)

    assert len(segs) == 1
    assert segs[0].shift == pytest.approx(1.2, abs=0.05)


def _duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize(
    "segments, expected",
    [
        # dub missing 1.5 s after 5 s -> silence padded in
        ([Segment(0.0, 5.0, 0.0), Segment(5.0, None, 1.5)], 21.5),
        # dub has 1 s extra after 5 s -> dropped
        ([Segment(0.0, 5.0, 0.0), Segment(5.0, None, -1.0)], 19.0),
    ],
)
def test_render_dub_pads_and_trims_cuts(tmp_path, segments, expected):
    rng = np.random.default_rng(49)
    src = tmp_path / "dub.wav"
    _write_wav(src, _dialogue(rng, 20.0, level=0.2))
    out = tmp_path / "out.mka"

    render_dub(src, out, 1.0, segments, acodec="flac")

    assert _duration(out) == pytest.approx(expected, abs=0.1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_render_dub_retimes_then_pads(tmp_path):
    rng = np.random.default_rng(50)
    src = tmp_path / "dub.wav"
    _write_wav(src, _dialogue(rng, 20.0, level=0.2))
    out = tmp_path / "out.mka"
    speed = 25 / 23.976

    # segment times are post-tempo: the slowed dub lasts 20 * speed seconds
    render_dub(
        src, out, 1 / speed,
        [Segment(0.0, 10.0, 0.0), Segment(10.0, None, 2.0)], acodec="flac",
    )

    assert _duration(out) == pytest.approx(20 * speed + 2.0, abs=0.15)
