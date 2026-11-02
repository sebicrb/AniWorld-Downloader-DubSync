"""Graft a dub audio track into a target video container losslessly.

Takes an archive-quality video (all its streams) plus a standalone dub audio
file and produces an output with the dub appended as a secondary,
language-tagged audio track. Everything is stream-copied (``-c copy``); the only
timing adjustment is a constant ``-itsoffset`` on the dub input, which stays
lossless. Container differences are reconciled by :func:`_finalize_episode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import re

import ffmpeg

from ....config import logger
from ...common.common import _finalize_episode, _run_ffmpeg_with_progress


def _count_audio_streams(path: Path) -> int:
    """Number of audio streams already in *path* (probe-based).

    Used to place the appended dub at the correct output audio index for the
    ``-metadata:s:a:N`` / ``-disposition:a:N`` specifiers. Raises rather than
    guessing, since a wrong index would mis-tag an archive file.
    """

    try:
        probe = ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        raise RuntimeError(f"Could not probe target video: {path}") from exc

    return sum(
        1 for s in probe.get("streams", []) if s.get("codec_type") == "audio"
    )


#: Marker in the title tag of every track DubSync grafts ("German Dub
#: (DubSync)", "German Dub (DubSync #2)", ...) -- how erase finds them.
DUBSYNC_TITLE_MARK = "DubSync"


def select_dubsync_tracks(streams: Iterable[dict], audio_code: str) -> List[int]:
    """Audio-relative indices (``0:a:N``) of the tracks DubSync added.

    A track counts when its title carries :data:`DUBSYNC_TITLE_MARK`, or --
    for files grafted before titles were written -- when it is an untitled
    *audio_code* track that is not the file's first audio track. The first
    audio track is never selected: it is the release's own original audio.
    """

    audio = [s for s in streams if s.get("codec_type") == "audio"]
    picked = []
    for idx, stream in enumerate(audio):
        tags = {k.lower(): v for k, v in (stream.get("tags") or {}).items()}
        title = str(tags.get("title") or "")
        lang = str(tags.get("language") or "").lower()
        if DUBSYNC_TITLE_MARK.lower() in title.lower():
            picked.append(idx)
        elif idx > 0 and not title and lang == audio_code.lower():
            picked.append(idx)
    return picked


_TRY_NUMBER_RE = re.compile(re.escape(DUBSYNC_TITLE_MARK) + r"\s*#(\d+)", re.I)


def _audio_streams(path) -> List[dict]:
    try:
        probe = ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        raise RuntimeError(f"Could not probe target video: {path}") from exc
    return [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]


def find_dubsync_tracks(path, audio_code: str) -> List[int]:
    """Probe *path* and return :func:`select_dubsync_tracks` for it."""

    return select_dubsync_tracks(_audio_streams(path), audio_code)


def next_try_number(streams: Iterable[dict], audio_code: str) -> int:
    """Number for the next grafted track: one past the highest existing try
    (an un-numbered DubSync track counts as try 1), 1 when there is none."""

    streams = [s for s in streams if s.get("codec_type") == "audio"]
    highest = 0
    for idx in select_dubsync_tracks(streams, audio_code):
        tags = {k.lower(): v for k, v in (streams[idx].get("tags") or {}).items()}
        m = _TRY_NUMBER_RE.search(str(tags.get("title") or ""))
        highest = max(highest, int(m.group(1)) if m else 1)
    return highest + 1


def dubsync_track_title(language_label: str, number: int) -> str:
    """Title for a grafted track; try 1 is un-numbered ("German Dub
    (DubSync)"), later tries carry their number ("... (DubSync #2)")."""

    suffix = DUBSYNC_TITLE_MARK if number <= 1 else f"{DUBSYNC_TITLE_MARK} #{number}"
    return f"{language_label} ({suffix})"


def strip_audio_tracks(target_video, audio_indices: Iterable[int], label: str = ""):
    """Remove the given audio tracks (``0:a:N`` indices) from *target_video*
    in place. Stream copy only; written to a temp file and atomically moved
    over the original, so an interrupted run leaves the file untouched."""

    target_video = Path(target_video)
    drop = set(audio_indices)
    num_audio = _count_audio_streams(target_video)
    keep = [i for i in range(num_audio) if i not in drop]
    if not keep:
        raise RuntimeError(
            f"refusing to remove every audio track from {target_video.name}"
        )

    inp = ffmpeg.input(str(target_video))
    temp_path = target_video.with_name(f"{target_video.stem}.dubsync.strip.mkv")
    node = ffmpeg.output(
        inp["v?"],
        *[inp[f"a:{i}"] for i in keep],
        inp["s?"],
        inp["d?"],
        inp["t?"],
        str(temp_path),
        c="copy",
    )
    try:
        _run_ffmpeg_with_progress(node, label=label)
        _finalize_episode(temp_path, target_video, label)
    finally:
        temp_path.unlink(missing_ok=True)
    return target_video


def remux_with_dub(
    target_video,
    dub_audio,
    output_path,
    offset: float = 0.0,
    audio_code: str = "deu",
    label: str = "",
    title: str = "",
):
    """Mux ``target_video`` + ``dub_audio`` -> ``output_path``.

    Stream order in the output is: all of the target's streams first, then the
    dub audio appended (``-map 0 -map 1:a``). The dub is tagged
    ``language=<audio_code>`` and has its default disposition cleared so players
    keep the original track primary. ``offset`` (seconds, may be negative)
    delays the dub via ``-itsoffset``. ``title`` (if given) becomes the dub
    track's title tag -- set in the second pass, since ffmpeg-python cannot
    emit two ``-metadata`` options for the same stream specifier.

    Writes to a temp MKV next to ``output_path`` then atomically finalises onto
    it, so an interrupted run never corrupts the archive file.
    """

    target_video = Path(target_video)
    dub_audio = Path(dub_audio)
    output_path = Path(output_path)

    num_audio = _count_audio_streams(target_video)

    video_in = ffmpeg.input(str(target_video))

    dub_input_kwargs = {}
    if offset:
        dub_input_kwargs["itsoffset"] = float(offset)
    dub_in = ffmpeg.input(str(dub_audio), **dub_input_kwargs)

    output_kwargs = {
        "c": "copy",
        f"metadata:s:a:{num_audio}": f"language={audio_code}",
        # Keep the original audio as the default track; the dub is secondary.
        f"disposition:a:{num_audio}": "0",
    }

    temp_path = output_path.with_name(f"{output_path.stem}.dubsync.new.mkv")
    temp_attach = output_path.with_name(f"{output_path.stem}.dubsync.attach.mkv")

    logger.debug(
        f"[DUBSYNC] muxing dub (offset={offset}s, tag={audio_code}, "
        f"stream a:{num_audio}) -> {output_path.name}"
    )

    # ffmpeg (observed on 8.1) badly mis-interleaves the second input's
    # packets in a multi-input stream copy: the dub lands hundreds of MB
    # later in the file than its timestamp peers (or entirely at the end when
    # attachments are mapped) -- players then find no dub audio in their
    # readahead window and play silence. So the mux is done in two passes:
    #
    # 1. Combine target streams (minus attachments, which worsen the skew)
    #    with the offset dub. The result is correct but poorly interleaved.
    # 2. Re-interleave in a single-input copy with -max_interleave_delta 0,
    #    which buffers until every stream is present and writes strictly by
    #    timestamp; the target's font attachments are re-added in the same
    #    pass (an attachments-only extra input carries no packets and cannot
    #    skew anything).
    node = ffmpeg.output(
        video_in["v?"],
        video_in["a?"],
        video_in["s?"],
        video_in["d?"],
        dub_in.audio,
        str(temp_path),
        **output_kwargs,
    )

    try:
        _run_ffmpeg_with_progress(node, label=label)

        logger.debug(f"[DUBSYNC] re-interleaving -> {output_path.name}")
        interleave_kwargs = {"c": "copy", "max_interleave_delta": "0"}
        if title:
            interleave_kwargs[f"metadata:s:a:{num_audio}"] = f"title={title}"
        interleave_node = ffmpeg.output(
            ffmpeg.input(str(temp_path)),
            ffmpeg.input(str(target_video))["t?"],
            str(temp_attach),
            **interleave_kwargs,
        )
        _run_ffmpeg_with_progress(interleave_node, label=label)
        temp_path.unlink(missing_ok=True)

        _finalize_episode(temp_attach, output_path, label)
    finally:
        temp_path.unlink(missing_ok=True)
        temp_attach.unlink(missing_ok=True)

    return output_path
