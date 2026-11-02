"""DubSync pipeline: match -> preflight -> extract -> align -> remux -> cleanup.

Three modes: ``add`` (default) skips files that already carry the dub
language, ``retry`` grafts another, separately titled track next to an
existing one ("new try"), ``erase`` strips the tracks DubSync added.

Orchestrates grafting a German dub (from an AniWorld/SerienStream source) onto a
folder of archive-quality video files. The dub's offset against each target is
detected automatically per episode (:mod:`align`); a manual ``offset`` overrides
detection, and low-confidence detections fall back to 0 and are flagged so the
user can verify them.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ....config import LANG_CODE_MAP, LANG_KEY_MAP, LANG_LABELS, logger
from ...common.common import check_downloaded
from .align import DEFAULT_MIN_CONFIDENCE, detect_offset, render_dub
from .extract import extract_dub_audio
from .matcher import MatchReport, match_directory
from .remux import (
    _audio_streams,
    dubsync_track_title,
    find_dubsync_tracks,
    next_try_number,
    remux_with_dub,
    strip_audio_tracks,
)

#: ``add``: skip files that already have the dub language; ``retry``: graft
#: another track anyway; ``erase``: remove DubSync's tracks, download nothing.
MODES = ("add", "retry", "erase")

#: Which DubSync tracks erase removes. The newest is the one appended last
#: (highest audio index) -- i.e. the latest "new try".
#: ``all``; ``older`` keeps the newest (the new try won); ``newest`` drops
#: only the newest (the new try lost).
ERASE_WHICH = ("all", "older", "newest")


def dubsync_env_defaults() -> dict:
    """DubSync defaults from ``ANIWORLD_DUBSYNC_*`` (see .env.example).

    The CLI exports its flags to these variables before calling, so both the
    CLI and the web queue worker resolve settings through this one helper.
    """

    offset_raw = os.getenv("ANIWORLD_DUBSYNC_OFFSET", "").strip()
    try:
        offset = float(offset_raw) if offset_raw else None
    except ValueError:
        logger.warning(f"[DUBSYNC] ignoring invalid ANIWORLD_DUBSYNC_OFFSET: {offset_raw!r}")
        offset = None

    return {
        "target_dir": os.getenv("ANIWORLD_DUBSYNC_TARGET_DIR", "").strip(),
        "offset": offset,
        "auto_align": os.getenv("ANIWORLD_DUBSYNC_AUTO_ALIGN", "1") != "0",
        "allow_resample": os.getenv("ANIWORLD_DUBSYNC_ALLOW_RESAMPLE", "0") == "1",
        "cleanup": os.getenv("ANIWORLD_DUBSYNC_CLEANUP", "0") == "1",
        "audio_language": os.getenv("ANIWORLD_DUBSYNC_AUDIO_LANG", "German Dub").strip()
        or "German Dub",
    }


@dataclass
class FileOutcome:
    """What happened to one target file during a run."""

    path: Path
    status: str  # "done" | "skipped" | "failed"
    detail: str = ""


def _resolve_audio_code(audio_language: str) -> str:
    """Map a language label ("German Dub") to its ISO code ("deu")."""

    key = next((k for k, v in LANG_LABELS.items() if v == audio_language), None)
    if key is None:
        return "deu"
    audio_enum, _ = LANG_KEY_MAP[key]
    return LANG_CODE_MAP.get(audio_enum) or "deu"


def _print_report(report: MatchReport, target_dir) -> None:
    """Human-readable match table (also the whole output of ``--dry-run``)."""

    print(f"\nDubSync match report for: {target_dir}")
    print(f"  matched:   {len(report.pairs)}")
    print(f"  unmatched: {len(report.unmatched_files)} (no episode number parsed)")
    print(f"  unpaired:  {len(report.unpaired_files)} (parsed, no source episode)")
    print(f"  missing:   {len(report.missing_sources)} (source episode, no local file)")

    if report.pairs:
        print("\n  Pairs:")
        for pf, episode in report.pairs:
            season = pf.season if pf.season is not None else "?"
            title = getattr(episode, "title_de", None) or getattr(
                episode, "title_en", ""
            )
            print(
                f"    S{season}E{pf.episode:02d}  {pf.path.name}"
                + (f"  <-  {title}" if title else "")
            )

    if report.unpaired_files:
        print("\n  Unpaired (parsed but no matching source episode):")
        for pf in report.unpaired_files:
            season = pf.season if pf.season is not None else "?"
            print(f"    S{season}E{pf.episode:02d}  {pf.path.name}")

    if report.unmatched_files:
        print("\n  Unmatched (could not parse an episode number):")
        for path in report.unmatched_files:
            print(f"    {path.name}")

    if report.missing_sources:
        print("\n  Missing (source episodes with no local file):")
        for season, number in report.missing_sources:
            season_str = season if season is not None else "?"
            print(f"    S{season_str}E{number:02d}")
    print()


def _align_dub(
    dub_tmp: Path,
    target: Path,
    allow_resample: bool,
    min_confidence: float,
) -> Tuple[float, Path, str]:
    """Detect the dub's offset against *target*, handling drift and cuts.

    Returns ``(offset, dub_path, note)`` -- *dub_path* differs from *dub_tmp*
    only when the dub was re-rendered (tempo corrected and/or re-assembled
    around act-break cuts) into a new temp file. *note* describes what
    alignment decided (kept in the outcome detail so low-confidence
    fallbacks are visible in the final report).
    """

    result = detect_offset(dub_tmp, target, find_cuts=True)

    if result.confidence < min_confidence:
        note = (
            f"low-confidence alignment (z={result.confidence:.1f} < "
            f"{min_confidence:.0f}); fell back to offset 0.00s - verify manually"
        )
        logger.warning(f"[DUBSYNC] {target.name}: {note}")
        return 0.0, dub_tmp, note

    cuts = len(result.segments) - 1 if result.has_cuts else 0
    drift_ok = result.has_drift and result.drift_confidence >= min_confidence

    fixes = []
    if drift_ok:
        fixes.append(
            f"drift {result.drift:+.2f}s via atempo={result.tempo_ratio:.5f}"
        )
    if cuts:
        jumps = ", ".join(
            f"{b.shift - a.shift:+.2f}s@{b.start / 60:.0f}m"
            for a, b in zip(result.segments, result.segments[1:])
        )
        fixes.append(f"{cuts} cut(s) bridged ({jumps})")

    if fixes and allow_resample:
        rendered = dub_tmp.with_name(f"{dub_tmp.stem}.retimed.mka")
        render_dub(
            dub_tmp,
            rendered,
            result.tempo_ratio if drift_ok else 1.0,
            result.segments if cuts else None,
            label=target.stem,
        )
        offset = (
            result.segments[0].shift
            if result.segments
            else (result.post_tempo_offset if drift_ok else result.offset)
        )
        note = (
            f"offset {offset:+.2f}s (z={result.confidence:.1f}); "
            + "; ".join(fixes)
            + " (dub re-encoded)"
        )
        return offset, rendered, note

    if fixes:
        note = (
            f"offset {result.offset:+.2f}s (z={result.confidence:.1f}); needs "
            + "; ".join(fixes)
            + " but speed/cut fixing is disabled - sync will wander"
        )
        logger.warning(f"[DUBSYNC] {target.name}: {note}")
        return result.offset, dub_tmp, note

    note = f"offset {result.offset:+.2f}s (z={result.confidence:.1f})"
    return result.offset, dub_tmp, note


def _dry_run_align(
    pairs,
    tmp_dir: Path,
    audio_language: str,
    audio_code: str,
    allow_resample: bool,
    min_confidence: float,
) -> None:
    """Extract each dub to temp and print its detected offset (no writes to
    any target file). This is the trust-building step before a real run."""

    print("  Detected offsets (dry-run, nothing written):")
    for pf, episode in pairs:
        target = pf.path
        dub_tmp = tmp_dir / f"{target.stem}.{audio_code}.mka"
        try:
            extract_dub_audio(
                episode, dub_tmp, audio_language=audio_language, label=target.stem
            )
            result = detect_offset(dub_tmp, target, find_cuts=True)
            line = f"offset {result.offset:+.3f}s  (z={result.confidence:.1f})"
            if result.confidence < min_confidence:
                line += "  LOW CONFIDENCE - would fall back to 0"
            else:
                if result.has_drift:
                    line += f"  drift {result.drift:+.2f}s"
                    if allow_resample:
                        line += f" -> atempo={result.tempo_ratio:.5f}"
                if result.has_cuts:
                    line += f"  {len(result.segments) - 1} cut(s): " + ", ".join(
                        f"{b.shift - a.shift:+.2f}s@{b.start:.0f}s"
                        for a, b in zip(result.segments, result.segments[1:])
                    )
                if (result.has_drift or result.has_cuts) and not allow_resample:
                    line += " (speed/cut fixing disabled)"
            print(f"    {target.name}: {line}")
        except Exception as exc:  # noqa: BLE001 - keep probing the rest
            print(f"    {target.name}: FAILED ({exc})")
        finally:
            dub_tmp.unlink(missing_ok=True)
    print()


def _print_summary(outcomes: List[FileOutcome]) -> None:
    done = sum(1 for o in outcomes if o.status == "done")
    skipped = sum(1 for o in outcomes if o.status == "skipped")
    failed = [o for o in outcomes if o.status == "failed"]
    print(
        f"\nDubSync finished: {done} done, {skipped} skipped, {len(failed)} failed"
    )
    for o in failed:
        print(f"  FAILED  {o.path.name}: {o.detail}")
    print()


def _erase_tracks(pairs, audio_code: str, which: str = "all") -> List[FileOutcome]:
    """Erase mode: strip DubSync's tracks from every matched file in place
    (*which*: see :data:`ERASE_WHICH`)."""

    outcomes: List[FileOutcome] = []
    for pf, _episode in pairs:
        target = pf.path
        try:
            tracks = find_dubsync_tracks(target, audio_code)
            if which == "older":
                tracks = tracks[:-1]
            elif which == "newest":
                tracks = tracks[-1:]
            if not tracks:
                outcomes.append(
                    FileOutcome(
                        target,
                        "skipped",
                        "no DubSync track to erase"
                        + (" (only one present, kept)" if which == "older" else ""),
                    )
                )
                continue
            strip_audio_tracks(target, tracks, label=target.stem)
            detail = f"removed {len(tracks)} track(s): " + ", ".join(
                f"a:{i}" for i in tracks
            )
            logger.info(f"[DUBSYNC] erased: {target.name} ({detail})")
            outcomes.append(FileOutcome(target, "done", detail))
        except Exception as exc:  # noqa: BLE001 - report per-file, keep going
            logger.warning(f"[DUBSYNC] erase failed for {target.name}: {exc}")
            outcomes.append(FileOutcome(target, "failed", str(exc)))
    return outcomes


def run_dubsync(
    source,
    target_dir,
    offset: Optional[float] = None,
    audio_language: str = "German Dub",
    recursive: bool = False,
    dry_run: bool = False,
    cleanup: bool = False,
    overwrite: bool = False,
    mode: str = "add",
    erase_which: str = "all",
    auto_align: bool = True,
    allow_resample: bool = False,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    selected: Optional[Iterable[Tuple[Optional[int], int]]] = None,
    explicit: Optional[Iterable[Tuple[Optional[int], int, str]]] = None,
) -> Tuple[MatchReport, List[FileOutcome]]:
    """Run DubSync over ``target_dir`` against ``source``.

    Args:
        source: AniWorld/SerienStream URL string, or a series/season/episode
            model object (anything the matcher accepts).
        target_dir: directory of archive-quality videos to enrich.
        offset: manual dub delay in seconds (negative allowed). ``None`` means
            detect per episode when ``auto_align`` is on (else 0); an explicit
            value always overrides detection.
        audio_language: source language label to graft (default German Dub).
        recursive: descend into subdirectories when scanning.
        dry_run: print the match report (and, with auto-align, the detected
            per-episode offsets) without writing to any target file.
        cleanup: edit files in place (temp + atomic replace) instead of writing
            a ``*.dubsync.<ext>`` sidecar copy.
        overwrite: legacy alias for ``mode="retry"``.
        mode: ``add`` skips files that already carry the dub language;
            ``retry`` grafts another, separately titled track next to any
            existing one; ``erase`` removes the tracks DubSync added (always
            in place, nothing is downloaded).
        erase_which: for ``erase``: ``all`` DubSync tracks, ``older`` (keep
            the newest try) or ``newest`` (drop only the newest try).
        auto_align: detect each episode's offset via music/SFX-bed correlation.
        allow_resample: when linear drift (e.g. PAL-sourced dub) or
            act-break cuts are detected, re-render the dub with ``atempo``
            and silence padded into the cuts -- re-encodes the dub track only.
        min_confidence: correlation peak z-score below which a detection is
            discarded in favour of offset 0 (and flagged in the report).
        selected: restrict the run to these ``(season, episode)`` keys
            (``None`` processes every episode the matcher pairs).
        explicit: user-confirmed ``(season, episode, filename)`` triples for
            movies, whose filenames carry no parsable episode pattern.

    Returns ``(match_report, outcomes)``.
    """

    if overwrite and mode == "add":
        mode = "retry"
    if mode not in MODES:
        raise ValueError(f"unknown DubSync mode {mode!r} (expected one of {MODES})")
    if erase_which not in ERASE_WHICH:
        raise ValueError(
            f"unknown erase selection {erase_which!r} (expected one of {ERASE_WHICH})"
        )

    report = match_directory(
        target_dir, source, recursive=recursive, selected=selected, explicit=explicit
    )
    audio_code = _resolve_audio_code(audio_language)

    _print_report(report, target_dir)

    outcomes: List[FileOutcome] = []
    use_auto_align = auto_align and offset is None

    if dry_run:
        if use_auto_align and report.pairs:
            tmp_dir = Path(tempfile.mkdtemp(prefix="dubsync_"))
            try:
                _dry_run_align(
                    report.pairs,
                    tmp_dir,
                    audio_language,
                    audio_code,
                    allow_resample,
                    min_confidence,
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("[DUBSYNC] dry-run: no files written")
        return report, outcomes

    if not report.pairs:
        logger.info("[DUBSYNC] nothing to do (no matched files)")
        return report, outcomes

    if mode == "erase":
        outcomes = _erase_tracks(report.pairs, audio_code, erase_which)
        _print_summary(outcomes)
        return report, outcomes

    tmp_dir = Path(tempfile.mkdtemp(prefix="dubsync_"))
    try:
        for pf, episode in report.pairs:
            target = pf.path
            try:
                check = check_downloaded(target)
                if audio_code in check["audio_langs"] and mode == "add":
                    logger.info(
                        f"[DUBSYNC] skip (already has '{audio_code}'): {target.name}"
                    )
                    outcomes.append(
                        FileOutcome(
                            target, "skipped", f"'{audio_code}' track already present"
                        )
                    )
                    continue

                dub_tmp = tmp_dir / f"{target.stem}.{audio_code}.mka"
                extract_dub_audio(
                    episode, dub_tmp, audio_language=audio_language, label=target.stem
                )

                detail = ""
                dub_path = dub_tmp
                if use_auto_align:
                    use_offset, dub_path, detail = _align_dub(
                        dub_tmp, target, allow_resample, min_confidence
                    )
                else:
                    use_offset = offset if offset is not None else 0.0

                if cleanup:
                    out = target
                else:
                    out = target.with_name(f"{target.stem}.dubsync{target.suffix}")

                number = next_try_number(_audio_streams(target), audio_code)
                remux_with_dub(
                    target,
                    dub_path,
                    out,
                    offset=use_offset,
                    audio_code=audio_code,
                    label=target.stem,
                    title=dubsync_track_title(audio_language, number),
                )

                if dub_path != dub_tmp:
                    dub_path.unlink(missing_ok=True)
                dub_tmp.unlink(missing_ok=True)
                outcomes.append(FileOutcome(out, "done", detail))
                logger.info(f"[DUBSYNC] done: {out.name}")

            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                logger.warning(f"[DUBSYNC] failed for {target.name}: {exc}")
                outcomes.append(FileOutcome(target, "failed", str(exc)))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _print_summary(outcomes)
    return report, outcomes
