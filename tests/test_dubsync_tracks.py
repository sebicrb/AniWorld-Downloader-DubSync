"""DubSync track bookkeeping: finding the tracks DubSync added (for erase and
for numbering a new try), stripping them, and the add/retry/erase modes."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aniworld.models.aniworld_to.dubsync import pipeline as pipeline_mod
from aniworld.models.aniworld_to.dubsync.matcher import MatchReport
from aniworld.models.aniworld_to.dubsync.remux import (
    dubsync_track_title,
    next_try_number,
    select_dubsync_tracks,
    strip_audio_tracks,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _a(lang=None, title=None):
    tags = {}
    if lang:
        tags["language"] = lang
    if title:
        tags["title"] = title
    return {"codec_type": "audio", "tags": tags}


def test_select_titled_tracks():
    streams = [
        {"codec_type": "video"},
        _a("eng"),
        _a("deu", "German Dub (DubSync)"),
        _a("deu", "German Dub (DubSync #2)"),
    ]
    assert select_dubsync_tracks(streams, "deu") == [1, 2]


def test_select_legacy_untitled_dub_but_never_first_track():
    # Grafted before titles existed: untitled deu that is not the first track.
    assert select_dubsync_tracks([_a("eng"), _a("deu")], "deu") == [1]
    # A German-only release keeps its own track.
    assert select_dubsync_tracks([_a("deu"), _a("eng")], "deu") == []


def test_select_keeps_release_tracks_with_titles():
    streams = [_a("eng", "English 5.1"), _a("deu", "Deutsch DD5.1"), _a("deu")]
    assert select_dubsync_tracks(streams, "deu") == [2]


def test_track_titles_number_from_second_try():
    assert dubsync_track_title("German Dub", 1) == "German Dub (DubSync)"
    assert dubsync_track_title("German Dub", 2) == "German Dub (DubSync #2)"


def test_next_try_number():
    assert next_try_number([_a("eng")], "deu") == 1
    # legacy untitled graft counts as try 1
    assert next_try_number([_a("eng"), _a("deu")], "deu") == 2
    # after "keep the newest" erased try 1, the next one is #3, not #2 again
    assert next_try_number([_a("eng"), _a("deu", "German Dub (DubSync #2)")], "deu") == 3


def _probe_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream_tags=language,title", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return [s.get("tags", {}) for s in json.loads(out)["streams"]]


def _make_mkv(path: Path, langs):
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc=size=64x64:rate=10:duration=2"]
    for i, _ in enumerate(langs):
        args += ["-f", "lavfi", "-i", f"sine=frequency={300 + 100 * i}:duration=2"]
    args += ["-map", "0:v"]
    for i, _ in enumerate(langs):
        args += ["-map", f"{i + 1}:a"]
    args += ["-c:v", "mpeg4", "-c:a", "aac"]
    for i, lang in enumerate(langs):
        args += [f"-metadata:s:a:{i}", f"language={lang}"]
    args.append(str(path))
    subprocess.run(args, check=True)


@needs_ffmpeg
def test_strip_audio_tracks_in_place(tmp_path):
    f = tmp_path / "ep.mkv"
    _make_mkv(f, ["eng", "deu", "deu"])

    strip_audio_tracks(f, [2])

    assert [t.get("language") for t in _probe_audio(f)] == ["eng", "deu"]
    assert not list(tmp_path.glob("*.strip.mkv"))


@needs_ffmpeg
def test_strip_refuses_to_remove_all_audio(tmp_path):
    f = tmp_path / "ep.mkv"
    _make_mkv(f, ["deu"])
    with pytest.raises(RuntimeError):
        strip_audio_tracks(f, [0])


class _PF:
    def __init__(self, path):
        self.path = path
        self.season = 1
        self.episode = 1


@needs_ffmpeg
def test_erase_mode_strips_only_dubsync_tracks(tmp_path, monkeypatch):
    f = tmp_path / "Show - S01E01.mkv"
    _make_mkv(f, ["eng", "deu"])  # legacy, untitled DubSync track
    plain = tmp_path / "Show - S01E02.mkv"
    _make_mkv(plain, ["eng"])

    report = MatchReport()
    report.pairs = [(_PF(f), object()), (_PF(plain), object())]
    monkeypatch.setattr(pipeline_mod, "match_directory", lambda *a, **k: report)
    monkeypatch.setattr(
        pipeline_mod, "extract_dub_audio",
        lambda *a, **k: pytest.fail("erase must not download anything"),
    )

    _, outcomes = pipeline_mod.run_dubsync("url", tmp_path, mode="erase")

    assert [o.status for o in outcomes] == ["done", "skipped"]
    assert [t.get("language") for t in _probe_audio(f)] == ["eng"]


@needs_ffmpeg
@pytest.mark.parametrize(
    "which, left",
    [("all", ["eng"]), ("older", ["eng", "deu"]), ("newest", ["eng", "deu"])],
)
def test_erase_selection(tmp_path, monkeypatch, which, left):
    f = tmp_path / "Show - S01E01.mkv"
    _make_mkv(f, ["eng", "deu", "deu"])  # two untitled (legacy) grafts

    report = MatchReport()
    report.pairs = [(_PF(f), object())]
    monkeypatch.setattr(pipeline_mod, "match_directory", lambda *a, **k: report)

    _, outcomes = pipeline_mod.run_dubsync(
        "url", tmp_path, mode="erase", erase_which=which
    )

    assert outcomes[0].status == "done"
    assert [t.get("language") for t in _probe_audio(f)] == left
    if which == "older":
        # the newest (last appended) track survives
        assert outcomes[0].detail.endswith("a:1")
    if which == "newest":
        assert outcomes[0].detail.endswith("a:2")


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        pipeline_mod.run_dubsync("url", tmp_path, mode="bogus")
