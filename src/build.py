"""Точка входа сборки звуковой дорожки.

Запуск::

    python src/build.py
    python src/build.py --scenario scenario/timeline.json
    python src/build.py --no-normalize
    python src/build.py --verbose

Сборка детерминирована: одинаковый сценарий и одинаковые исходники дают
одинаковый результат. Исходные WAV-файлы только читаются.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    ScenarioError,
    Timeline,
    configure_console,
    format_timecode,
    load_timeline,
)
from renderer import (
    LoudnessResult,
    RenderContext,
    RenderError,
    StemResult,
    envelope_summary,
    expected_level,
    render_all,
    write_command_log,
)
from validator import ValidationReport, print_report, validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rel(path: Path) -> str:
    """Путь относительно корня проекта; для внешних путей — как есть.

    Обычно все пути лежат внутри проекта, но --scenario и --output допускают
    абсолютный путь куда угодно, и relative_to() на нём падает.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Детерминированная сборка звуковой дорожки косплей-выступления.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python src/build.py\n"
            "  python src/build.py --scenario scenario/timeline.json\n"
            "  python src/build.py --no-normalize\n"
            "  python src/build.py --verbose\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        default="scenario/timeline.json",
        help="путь к файлу сценария относительно корня проекта "
        "(по умолчанию: scenario/timeline.json)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="каталог для результатов (по умолчанию: output)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="не приводить громкость к целевым LUFS; применить только лимитер True Peak",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="показывать все команды ffmpeg и их вывод",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="выполнить только проверки, не запускать рендер",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="не удалять промежуточный микс output/_premaster.wav",
    )
    return parser.parse_args(argv)


def _stem_to_dict(stem: StemResult) -> dict:
    return {
        "name": stem.name,
        "file": rel(stem.path),
        "silent": stem.silent,
        "duration_seconds": round(stem.duration, 3) if stem.duration else None,
        "peak_dbfs": stem.peak_db,
        "events": stem.event_ids,
        "skipped_events": stem.skipped_event_ids,
    }


def _loudness_to_dict(loudness: LoudnessResult) -> dict:
    def measured(block: dict | None) -> dict | None:
        """loudnorm печатает числа строками — приводим к float для отчёта."""
        if not block:
            return None
        names = {
            "input_i": "integrated_lufs",
            "input_tp": "true_peak_db",
            "input_lra": "lra_lu",
            "input_thresh": "threshold_lufs",
        }
        result: dict[str, float | None] = {}
        for key, name in names.items():
            if key not in block:
                continue
            try:
                result[name] = float(block[key])
            except (TypeError, ValueError):
                result[name] = None
        return result

    return {
        "normalization_applied": loudness.applied,
        "target_lufs": loudness.target_lufs,
        "effective_target_lufs": loudness.effective_target_lufs,
        "requested_gain_db": loudness.requested_gain_db,
        "applied_gain_db": loudness.applied_gain_db,
        "gain_capped": loudness.gain_capped,
        "cap_reason": loudness.cap_reason,
        # "linear" — один коэффициент на всю дорожку, динамика сценария
        # сохранена; "dynamic" — loudnorm переформовал динамику.
        "normalization_type": loudness.normalization_type,
        "max_linear_gain_db": loudness.max_linear_gain_db,
        "measured_before": measured(loudness.measured_input),
        "measured_after": measured(loudness.measured_output),
    }


def build_report(
    args: argparse.Namespace,
    timeline: Timeline,
    validation: ValidationReport,
    result: dict,
    ctx: RenderContext,
) -> dict:
    """Формирует содержимое render-report.json."""
    duck_intervals = result["duck_intervals"]
    ducking = timeline.ducking

    events_report = []
    for event in sorted(result["usable_events"], key=lambda e: (e.start, e.id)):
        start, end = event.window()
        events_report.append(
            {
                "id": event.id,
                "stem": event.stem,
                "file": event.file.as_posix(),
                "start": format_timecode(start),
                "end": format_timecode(end),
                "duration_seconds": round(end - start, 3),
                "source_duration_seconds": round(event.source_duration or 0.0, 3),
                "looped": bool(event.loop and (event.source_duration or 0) < end - start),
                "pan": event.pan,
                "duck_group": event.duck_group,
                # dB в сценарии — относительное ослабление, а не абсолютная
                # цель, поэтому показываем и уровень исходника, и расчётный
                # уровень события в миксе.
                "level": expected_level(event, ctx, duck_intervals),
                "envelope_db": envelope_summary(event, duck_intervals, ducking),
                "note": event.note,
            }
        )

    missing_optional = [
        {
            "id": event.id,
            "stem": event.stem,
            "file": event.file.as_posix(),
            "expected_start": format_timecode(event.start),
            "reason": reason,
        }
        for event, reason in result["skipped_events"]
    ]

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "project_root": PROJECT_ROOT.name,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ffmpeg": validation.ffmpeg_version,
        },
        "scenario": {
            "file": rel(timeline.source_path),
            "total_duration_seconds": timeline.total_duration,
            "total_duration_timecode": format_timecode(timeline.total_duration),
            "stems": timeline.stems,
        },
        "options": {
            "normalize": not args.no_normalize,
            "verbose": args.verbose,
            "keep_intermediate": args.keep_intermediate,
        },
        "output": {
            "sample_rate": timeline.output.sample_rate,
            "channels": timeline.output.channels,
            "bit_depth": timeline.output.bit_depth,
            "codec": timeline.output.pcm_codec,
            "target_lufs": timeline.output.target_lufs,
            "target_true_peak_db": timeline.output.target_true_peak,
            "true_peak_margin_db": timeline.output.true_peak_margin_db,
            "max_normalize_gain_db": timeline.output.max_normalize_gain_db,
        },
        "master": {
            "file": rel(result["master"]["path"]),
            "duration_seconds": round(result["master"]["duration"], 3)
            if result["master"]["duration"]
            else None,
            "sample_rate": result["master"]["sample_rate"],
            "channels": result["master"]["channels"],
            "codec": result["master"]["codec"],
            "peak_dbfs": result["master"]["peak_db"],
        },
        "loudness": _loudness_to_dict(result["loudness"]),
        "ducking": {
            "attack_seconds": ducking.attack,
            "release_seconds": ducking.release,
            "depth_db": ducking.groups,
            "voice_windows": [
                {"start": format_timecode(a), "end": format_timecode(b)}
                for a, b in duck_intervals
            ],
            "method": "предрасчитанная volume-автоматизация по таймкодам голосов "
            "(без sidechaincompress)",
        },
        "looping": {
            "crossfade_seconds": timeline.loop.crossfade,
            "curve": timeline.loop.curve,
            "method": "equal-power acrossfade между копиями файла",
        },
        "transition": {
            "fade_end": format_timecode(timeline.transition.fade_end),
            "gap_ms": timeline.transition.gap_ms,
            "ice_event_id": timeline.transition.ice_event_id,
            "note": timeline.transition.note,
        },
        "stems": [_stem_to_dict(stem) for stem in result["stems"]],
        "events": events_report,
        "missing_optional_assets": missing_optional,
        "validation": {
            "errors": validation.errors,
            "warnings": validation.warnings,
        },
        "render_warnings": ctx.warnings,
        "ffmpeg_command_count": len(ctx.commands),
    }


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)

    scenario_path = PROJECT_ROOT / args.scenario
    output_dir = PROJECT_ROOT / args.output

    print("=" * 72)
    print("Сборка звуковой дорожки")
    print("=" * 72)
    print(f"Корень проекта : {PROJECT_ROOT}")
    print(f"Сценарий       : {args.scenario}")
    print(f"Результаты     : {args.output}")
    print()

    print("[1/4] Чтение сценария")
    try:
        timeline = load_timeline(scenario_path, PROJECT_ROOT)
    except ScenarioError as exc:
        print(f"  [ERROR] {exc}")
        return 2
    print(
        f"  Общая длительность: {format_timecode(timeline.total_duration)} "
        f"({timeline.total_duration:g} с), событий: {len(timeline.events)}"
    )
    print()

    print("[2/4] Валидация")
    validation = validate(timeline, PROJECT_ROOT)
    print_report(validation)
    print()
    if not validation.ok:
        print("Сборка прервана: устраните ошибки выше.")
        return 1
    if args.validate_only:
        print("Указан --validate-only: рендер не запускался.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RenderContext(
        timeline=timeline, project_root=PROJECT_ROOT, verbose=args.verbose
    )
    ctx.probes.update(validation.probes)

    print("[3/4] Рендер")
    try:
        result = render_all(
            ctx,
            output_dir,
            normalize=not args.no_normalize,
            keep_intermediate=args.keep_intermediate,
        )
    except RenderError as exc:
        print(f"  [ERROR] {exc}")
        write_command_log(ctx, output_dir / "ffmpeg-command.txt")
        print(f"  Команды сохранены в {args.output}/ffmpeg-command.txt для разбора.")
        return 1
    print()

    print("[4/4] Отчёты")
    report = build_report(args, timeline, validation, result, ctx)
    report_path = output_dir / "render-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    command_path = output_dir / "ffmpeg-command.txt"
    write_command_log(ctx, command_path)
    print(f"  {rel(report_path)}")
    print(f"  {rel(command_path)}")
    print()

    _print_summary(report)
    return 0


def _format_peak(peak: float | None) -> str:
    """astats возвращает None для ровной тишины — показываем это явно."""
    return "тишина" if peak is None else f"пик {peak:+.2f} dBFS"


def _print_summary(report: dict) -> None:
    print("-" * 72)
    print("Готовые файлы")
    print("-" * 72)
    master = report["master"]
    print(
        f"  {master['file']:<28} {master['duration_seconds']} с  "
        f"{master['sample_rate']} Гц  {master['channels']} кан.  "
        f"{master['codec']}  {_format_peak(master['peak_dbfs'])}"
    )
    for stem in report["stems"]:
        tag = "тишина" if stem["silent"] else f"{len(stem['events'])} соб."
        print(
            f"  {stem['file']:<28} {stem['duration_seconds']} с  "
            f"{_format_peak(stem['peak_dbfs']):<18} ({tag})"
        )
    print(f"  {report['scenario']['file']:<28} (сценарий, вход)")
    print()

    loudness = report["loudness"]
    if loudness["measured_after"]:
        print(
            f"  Громкость master: "
            f"{loudness['measured_after']['integrated_lufs']:.1f} LUFS, "
            f"True Peak {loudness['measured_after']['true_peak_db']:.2f} dB "
            f"(цель {loudness['target_lufs']:.1f} LUFS / "
            f"{report['output']['target_true_peak_db']:.1f} dB)"
        )
        if loudness["gain_capped"]:
            print(
                f"  Прирост ограничен: запрошено {loudness['requested_gain_db']:+.1f} dB, "
                f"применено {loudness['applied_gain_db']:+.1f} dB"
            )
    print()

    missing = report["missing_optional_assets"]
    if missing:
        print("-" * 72)
        print(f"Отсутствующие необязательные ассеты ({len(missing)})")
        print("-" * 72)
        for item in missing:
            print(f"  {item['expected_start']}  {item['stem']:<9} {item['file']}")
        print()
        print("  Эти события пропущены. Положите файлы на место и запустите сборку")
        print("  заново — таймкоды и приглушение подхватятся автоматически.")
    else:
        print("  Все ассеты сценария найдены.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
