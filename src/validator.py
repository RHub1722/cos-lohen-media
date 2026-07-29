"""Предрендерная валидация: окружение, ассеты и сам таймлайн.

Валидатор разделяет находки на три уровня:

* ``errors``   — сборка невозможна (нет FFmpeg, битый обязательный WAV, ...);
* ``warnings`` — сборка возможна, но результат неполный (нет голосов, ...);
* ``info``     — просто справка для отчёта.

Модуль можно запустить отдельно::

    python src/validator.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python src/validator.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    Event,
    ScenarioError,
    Timeline,
    configure_console,
    format_timecode,
    load_timeline,
)

REQUIRED_DIRS = (
    "assets/ambience",
    "assets/voices",
    "assets/music",
    "assets/combat",
    "assets/ice",
    "scenario",
    "output",
)

# Допуск на сравнение таймкодов: 0.5 мс, меньше одного сэмпла на 48 кГц не нужно.
EPS = 5e-4


@dataclass
class ProbeResult:
    """Результат ffprobe по одному файлу."""

    path: Path
    exists: bool
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    error: str | None = None


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    probes: dict[str, ProbeResult] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    ffmpeg_version: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "ffmpeg_version": self.ffmpeg_version,
        }


def _tool_version(tool: str) -> str | None:
    """Возвращает первую строку ``<tool> -version`` или None, если его нет."""
    if shutil.which(tool) is None:
        return None
    try:
        proc = subprocess.run(
            [tool, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").splitlines()[0].strip() if proc.stdout else None


def probe_file(path: Path, project_root: Path) -> ProbeResult:
    """Читает параметры аудиофайла через ffprobe. Не изменяет исходник."""
    absolute = project_root / path
    if not absolute.is_file():
        return ProbeResult(path=path, exists=False)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(path=path, exists=True, error=f"ffprobe не запустился: {exc}")

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return ProbeResult(
            path=path,
            exists=True,
            error=detail[-1] if detail else f"ffprobe завершился с кодом {proc.returncode}",
        )

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return ProbeResult(path=path, exists=True, error=f"нечитаемый вывод ffprobe: {exc}")

    streams = data.get("streams") or []
    if not streams:
        return ProbeResult(path=path, exists=True, error="в файле нет аудиопотока")

    stream = streams[0]
    duration_raw = (data.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    return ProbeResult(
        path=path,
        exists=True,
        duration=duration,
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=int(stream["channels"]) if stream.get("channels") else None,
        codec=stream.get("codec_name"),
    )


def _check_environment(report: ValidationReport, project_root: Path) -> None:
    ffmpeg_version = _tool_version("ffmpeg")
    if ffmpeg_version is None:
        report.errors.append(
            "FFmpeg не найден в PATH. Установите его и убедитесь, что "
            "'ffmpeg -version' работает (см. README, раздел 1)."
        )
    else:
        report.ffmpeg_version = ffmpeg_version
        report.info.append(ffmpeg_version)

    if _tool_version("ffprobe") is None:
        report.errors.append(
            "ffprobe не найден в PATH. Он входит в тот же дистрибутив, что и FFmpeg."
        )

    if sys.version_info < (3, 11):
        report.errors.append(
            f"Требуется Python 3.11 или новее, запущен {sys.version.split()[0]}."
        )

    for rel in REQUIRED_DIRS:
        directory = project_root / rel
        if not directory.is_dir():
            report.errors.append(f"отсутствует обязательная папка: {rel}")


def _check_timeline_structure(report: ValidationReport, timeline: Timeline) -> None:
    total = timeline.total_duration
    if total <= 0:
        report.errors.append(f"total_duration={total} должен быть больше нуля")

    seen: dict[str, int] = {}
    for event in timeline.events:
        if event.id in seen:
            report.errors.append(
                f"дублирующийся идентификатор события: {event.id!r} "
                f"(встречается минимум дважды)"
            )
        seen[event.id] = seen.get(event.id, 0) + 1

        if event.start < -EPS:
            report.errors.append(
                f"{event.id}: отрицательный таймкод start={format_timecode(event.start)}"
            )
        if event.start > total + EPS:
            report.errors.append(
                f"{event.id}: start={format_timecode(event.start)} выходит за пределы "
                f"{format_timecode(total)}"
            )
        if event.end is not None:
            if event.end < -EPS:
                report.errors.append(
                    f"{event.id}: отрицательный таймкод end={format_timecode(event.end)}"
                )
            if event.end <= event.start + EPS:
                report.errors.append(
                    f"{event.id}: end={format_timecode(event.end)} не позже "
                    f"start={format_timecode(event.start)}"
                )
            if event.end > total + EPS:
                report.errors.append(
                    f"{event.id}: end={format_timecode(event.end)} выходит за пределы "
                    f"{format_timecode(total)}"
                )

        if event.stem not in timeline.stems:
            report.errors.append(
                f"{event.id}: stem={event.stem!r} отсутствует в списке stems "
                f"{timeline.stems}"
            )

        if event.duck_group and event.duck_group not in timeline.ducking.groups:
            report.warnings.append(
                f"{event.id}: duck_group={event.duck_group!r} не описана в ducking.groups — "
                f"приглушение для этого события не применяется"
            )

        for point in event.volume_points:
            if point.t < event.start - EPS:
                report.errors.append(
                    f"{event.id}: точка громкости {format_timecode(point.t)} раньше начала "
                    f"события {format_timecode(event.start)}"
                )
            if point.t > total + EPS:
                report.errors.append(
                    f"{event.id}: точка громкости {format_timecode(point.t)} выходит за "
                    f"пределы {format_timecode(total)}"
                )
            if event.end is not None and point.t > event.end + EPS:
                report.errors.append(
                    f"{event.id}: точка громкости {format_timecode(point.t)} позже конца "
                    f"события {format_timecode(event.end)}"
                )


def _check_transition(report: ValidationReport, timeline: Timeline) -> None:
    """Проверяет, что между затуханием зала и льдом реально есть пауза."""
    transition = timeline.transition
    ice = next((e for e in timeline.events if e.id == transition.ice_event_id), None)
    if ice is None:
        report.warnings.append(
            f"transition.ice_event_id={transition.ice_event_id!r}: событие не найдено, "
            f"переход не проверен"
        )
        return

    expected_start = transition.fade_end + transition.gap_ms / 1000.0
    if ice.start < expected_start - EPS:
        report.errors.append(
            f"переход: {ice.id} начинается в {format_timecode(ice.start)}, но затухание "
            f"зала заканчивается в {format_timecode(transition.fade_end)} и требуется "
            f"пауза {transition.gap_ms} мс — минимум {format_timecode(expected_start)}"
        )
    else:
        report.info.append(
            f"переход: пауза {round((ice.start - transition.fade_end) * 1000)} мс между "
            f"{format_timecode(transition.fade_end)} и {format_timecode(ice.start)}"
        )


def _check_assets(report: ValidationReport, timeline: Timeline, project_root: Path) -> None:
    if _tool_version("ffprobe") is None:
        report.warnings.append("ffprobe недоступен — проверка WAV-файлов пропущена")
        return

    expected_rate = timeline.output.sample_rate
    for event in timeline.events:
        key = event.file.as_posix()
        if key not in report.probes:
            report.probes[key] = probe_file(event.file, project_root)
        probe = report.probes[key]

        if not probe.exists:
            if event.required:
                report.missing_required.append(key)
                report.errors.append(
                    f"{event.id}: отсутствует обязательный файл {key}"
                )
            else:
                if key not in report.missing_optional:
                    report.missing_optional.append(key)
                report.warnings.append(
                    f"{event.id}: необязательный файл {key} не найден — событие пропущено"
                )
            continue

        if probe.error:
            message = f"{event.id}: файл {key} не читается — {probe.error}"
            if event.required:
                report.errors.append(message)
            else:
                report.warnings.append(f"{message} (событие пропущено)")
            continue

        if probe.duration is None or probe.duration <= 0:
            report.errors.append(f"{event.id}: у файла {key} нулевая длительность")
            continue

        if probe.sample_rate and probe.sample_rate != expected_rate:
            report.warnings.append(
                f"{event.id}: {key} имеет {probe.sample_rate} Гц вместо {expected_rate} Гц — "
                f"будет выполнено пересэмплирование"
            )
        if probe.channels and probe.channels != timeline.output.channels:
            report.warnings.append(
                f"{event.id}: {key} имеет {probe.channels} канал(ов) вместо "
                f"{timeline.output.channels} — раскладка будет приведена к "
                f"{timeline.output.channel_layout}"
            )

        _check_asset_duration(report, timeline, event, probe.duration)


def _check_asset_duration(
    report: ValidationReport, timeline: Timeline, event: Event, duration: float
) -> None:
    """Сверяет фактическую длину файла с окном события."""
    total = timeline.total_duration

    if event.end is not None:
        need = event.end - event.start
        if duration + EPS < need and not event.loop:
            report.warnings.append(
                f"{event.id}: файл короче окна события "
                f"({duration:.3f} с против {need:.3f} с) и loop=false — "
                f"остаток будет заполнен тишиной"
            )
        return

    # Длительность задаётся файлом (обычно голоса) — проверяем выход за 103 с.
    end = event.start + duration
    if end > total + EPS:
        report.warnings.append(
            f"{event.id}: файл длиной {duration:.3f} с, начинаясь в "
            f"{format_timecode(event.start)}, заканчивается в {format_timecode(end)} и "
            f"выходит за пределы {format_timecode(total)} — будет обрезан"
        )

    if event.slot_hint:
        slot_start, slot_end = event.slot_hint
        if end > slot_end + EPS:
            report.warnings.append(
                f"{event.id}: реплика заканчивается в {format_timecode(end)}, а ожидаемый "
                f"слот — {format_timecode(slot_start)}-{format_timecode(slot_end)}; "
                f"проверьте таймкоды в scenario/timeline.json"
            )


def _check_voice_overlaps(report: ValidationReport, timeline: Timeline) -> None:
    """Предупреждает о наложении реплик друг на друга."""
    voices: list[tuple[str, float, float]] = []
    for event in timeline.events:
        if event.stem != "voices":
            continue
        probe = report.probes.get(event.file.as_posix())
        if probe is None or not probe.exists or probe.duration is None or probe.error:
            continue
        voices.append((event.id, event.start, event.start + probe.duration))

    voices.sort(key=lambda item: item[1])
    for (id_a, _, end_a), (id_b, start_b, _) in zip(voices, voices[1:]):
        if start_b < end_a - EPS:
            report.warnings.append(
                f"реплики {id_a} и {id_b} накладываются на "
                f"{(end_a - start_b):.3f} с — проверьте таймкоды"
            )


def validate(timeline: Timeline, project_root: Path) -> ValidationReport:
    """Полная предрендерная проверка."""
    report = ValidationReport()
    _check_environment(report, project_root)
    _check_timeline_structure(report, timeline)
    _check_transition(report, timeline)
    _check_assets(report, timeline, project_root)
    _check_voice_overlaps(report, timeline)
    return report


def print_report(report: ValidationReport) -> None:
    """Печатает результат валидации в человекочитаемом виде."""
    for line in report.info:
        print(f"  [info]  {line}")
    for line in report.warnings:
        print(f"  [warn]  {line}")
    for line in report.errors:
        print(f"  [ERROR] {line}")
    if report.ok and not report.warnings:
        print("  Валидация пройдена без замечаний.")
    elif report.ok:
        print(f"  Валидация пройдена: предупреждений — {len(report.warnings)}.")
    else:
        print(f"  Валидация не пройдена: ошибок — {len(report.errors)}.")


def main() -> int:
    configure_console()
    project_root = Path(__file__).resolve().parent.parent
    scenario = project_root / "scenario" / "timeline.json"
    print(f"Сценарий: {scenario.relative_to(project_root).as_posix()}")
    try:
        timeline = load_timeline(scenario, project_root)
    except ScenarioError as exc:
        print(f"  [ERROR] {exc}")
        return 1
    report = validate(timeline, project_root)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
