"""Сборка filter_complex для FFmpeg и запуск рендера.

Всё микширование делает FFmpeg. Python только считает числа и собирает строки:

* длительности и число копий для бесшовного зацикливания;
* dB-огибающие громкости (ключевые точки сценария + приглушение под голосами);
* итоговые команды ffmpeg.

Исходные WAV-файлы открываются только на чтение и никогда не перезаписываются.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    SILENCE_DB,
    DuckingConfig,
    Event,
    Timeline,
    VolumePoint,
    format_timecode,
)
from validator import ProbeResult, probe_file

EPS = 5e-4

# Размер блока перед автоматизацией громкости. Фильтр volume пересчитывает
# выражение один раз на кадр, поэтому мелкий кадр = плавная огибающая
# (128 сэмплов при 48 кГц ≈ 2.7 мс) без «зиппера» на 80 мс атаке.
ENVELOPE_FRAME_SAMPLES = 128


class RenderError(Exception):
    """Рендер невозможно выполнить."""


@dataclass
class CommandRecord:
    """Одна выполненная команда ffmpeg — для ffmpeg-command.txt."""

    stage: str
    argv: list[str]

    def as_text(self) -> str:
        return " ".join(_quote(a) for a in self.argv)


@dataclass
class StemResult:
    name: str
    path: Path
    event_ids: list[str]
    skipped_event_ids: list[str]
    silent: bool
    duration: float | None = None
    peak_db: float | None = None


@dataclass
class LoudnessResult:
    applied: bool
    measured_input: dict | None = None
    measured_output: dict | None = None
    target_lufs: float | None = None
    effective_target_lufs: float | None = None
    requested_gain_db: float | None = None
    applied_gain_db: float | None = None
    gain_capped: bool = False
    cap_reason: str | None = None
    # Что loudnorm сделал фактически: "linear" (один коэффициент, динамика
    # сценария сохранена) или "dynamic" (переформовал динамику).
    normalization_type: str | None = None
    max_linear_gain_db: float | None = None


@dataclass
class RenderContext:
    """Состояние одного прогона сборки."""

    timeline: Timeline
    project_root: Path
    verbose: bool = False
    commands: list[CommandRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    probes: dict[str, ProbeResult] = field(default_factory=dict)
    # Пиковый уровень каждого исходника в dBFS — нужен, чтобы посчитать,
    # какой уровень событие получит в миксе. Ключ — путь в posix-виде.
    source_peaks: dict[str, float | None] = field(default_factory=dict)

    def probe(self, path: Path) -> ProbeResult:
        key = path.as_posix()
        if key not in self.probes:
            self.probes[key] = probe_file(path, self.project_root)
        return self.probes[key]

    def warn(self, message: str) -> None:
        # Часть расчётов вызывается и при рендере, и при сборке отчёта,
        # поэтому одно и то же предупреждение не должно попадать в отчёт дважды.
        if message in self.warnings:
            return
        self.warnings.append(message)
        print(f"  [warn]  {message}")


def _quote(arg: str) -> str:
    """Кавычит аргумент так, чтобы строку можно было прочитать глазами."""
    if arg and not re.search(r"[\s\"'|<>&^%()\[\]{}$;`]", arg):
        return arg
    return '"' + arg.replace('"', '\\"') + '"'


def _sanitize_label(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def arg_path(path: Path, project_root: Path) -> str:
    """Путь для командной строки ffmpeg — относительный от корня проекта.

    ffmpeg всегда запускается с cwd=project_root, поэтому относительные пути
    разрешаются корректно, а ffmpeg-command.txt остаётся переносимым: команды
    из него можно скопировать и выполнить на другой машине.
    """
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def db_to_amp(db: float) -> float:
    """Переводит dB в линейную амплитуду. Ниже SILENCE_DB — ровный ноль."""
    if db <= SILENCE_DB + EPS:
        return 0.0
    return 10.0 ** (db / 20.0)


# --------------------------------------------------------------------------- #
# Огибающие громкости
# --------------------------------------------------------------------------- #


def interpolate_db(points: list[VolumePoint], t: float) -> float:
    """Линейная интерполяция в dB. За пределами — удержание крайнего значения."""
    if not points:
        return 0.0
    if t <= points[0].t:
        return points[0].db
    if t >= points[-1].t:
        return points[-1].db
    for left, right in zip(points, points[1:]):
        if left.t <= t <= right.t:
            span = right.t - left.t
            if span <= EPS:
                return right.db
            ratio = (t - left.t) / span
            return left.db + (right.db - left.db) * ratio
    return points[-1].db


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Объединяет пересекающиеся интервалы в непересекающиеся."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + EPS:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _duck_contribution(
    t: float, interval: tuple[float, float], depth: float, attack: float, release: float
) -> float:
    """Смещение громкости в dB от одного голосового интервала в момент t."""
    start, end = interval
    if t <= start or depth <= 0:
        return 0.0
    if t < start + attack:
        return -depth * (t - start) / attack if attack > EPS else -depth
    if t <= end:
        return -depth
    if t < end + release:
        return -depth * (1.0 - (t - end) / release) if release > EPS else 0.0
    return 0.0


def duck_offset_at(
    t: float,
    intervals: list[tuple[float, float]],
    depth: float,
    ducking: DuckingConfig,
) -> float:
    """Итоговое смещение в dB: берём самое глубокое приглушение из активных."""
    if depth <= 0 or not intervals:
        return 0.0
    return min(
        _duck_contribution(t, interval, depth, ducking.attack, ducking.release)
        for interval in intervals
    )


def duck_breakpoints(
    intervals: list[tuple[float, float]], ducking: DuckingConfig
) -> list[float]:
    """Времена изломов кривой приглушения."""
    times: list[float] = []
    for start, end in intervals:
        times.extend([start, start + ducking.attack, end, end + ducking.release])
    return times


def build_envelope(
    event: Event,
    duck_intervals: list[tuple[float, float]],
    ducking: DuckingConfig,
) -> list[VolumePoint]:
    """Собирает единую огибающую события в ЛОКАЛЬНОМ времени (0 = начало события).

    Огибающая = ключевые точки сценария + постоянный gain_db + приглушение
    под голосами. Все три составляющие кусочно-линейны в dB, поэтому их сумма
    на объединённом наборе изломов вычисляется точно.
    """
    start, end = event.window()
    depth = ducking.depth_for(event.duck_group)

    base = list(event.volume_points)
    candidates = {start, end}
    candidates.update(p.t for p in base)
    if depth > 0:
        candidates.update(duck_breakpoints(duck_intervals, ducking))

    times = sorted(t for t in candidates if start - EPS <= t <= end + EPS)
    if not times:
        times = [start, end]

    points: list[VolumePoint] = []
    for t in times:
        clamped = min(max(t, start), end)
        db = event.gain_db
        db += interpolate_db(base, clamped) if base else 0.0
        db += duck_offset_at(clamped, duck_intervals, depth, ducking)
        local = round(clamped - start, 6)
        if points and abs(points[-1].t - local) <= EPS:
            # Совпавшие по времени точки: оставляем более тихую, чтобы
            # скачок вниз (например, обрыв фейда) не терялся.
            if db < points[-1].db:
                points[-1] = VolumePoint(t=local, db=db)
            continue
        points.append(VolumePoint(t=local, db=db))

    return points


def envelope_to_expr(points: list[VolumePoint]) -> str:
    """Превращает dB-огибающую в выражение для фильтра ``volume``.

    Результат — вложенная цепочка ``if(lt(t,T),...)``, линейная по dB внутри
    каждого сегмента. Выражение детерминировано: одинаковый вход даёт побитово
    одинаковый выход при каждом запуске.
    """
    if not points:
        return "1.0"
    if len(points) == 1:
        return f"{db_to_amp(points[0].db):.9f}"

    expr = f"{db_to_amp(points[-1].db):.9f}"
    for left, right in reversed(list(zip(points, points[1:]))):
        if right.t - left.t <= EPS:
            continue
        if abs(right.db - left.db) <= 1e-6:
            segment = f"{db_to_amp(left.db):.9f}"
        else:
            # 9 знаков: ошибка округления получается заметно ниже шага
            # 24-битного кванта (2^-23 ≈ 1.2e-7 от полной шкалы).
            slope = (right.db - left.db) / (right.t - left.t)
            segment = f"pow(10,({left.db:.9f}+{slope:.9f}*(t-{left.t:.6f}))/20)"
        expr = f"if(lt(t,{right.t:.6f}),{segment},{expr})"

    if points[0].t > EPS:
        expr = f"if(lt(t,{points[0].t:.6f}),{db_to_amp(points[0].db):.9f},{expr})"
    return expr


def source_compensation_db(event: Event, ctx: RenderContext) -> float:
    """Добавка, превращающая dB сценария в абсолютную цель.

    При ``normalize_source=true`` исходник приводится к пику 0 dBFS, поэтому
    значение -6 dB в сценарии означает «звучать на -6 dBFS», а не «ослабить
    файл на 6 dB». Компенсация складывается с огибающей в том же фильтре
    ``volume``, так что промежуточного сигнала на 0 dBFS не возникает.

    Побочный плюс: при замене WAV на файл другой громкости компенсация
    пересчитается сама, править сценарий не нужно.
    """
    if not event.normalize_source:
        return 0.0
    peak = ctx.source_peaks.get(event.file.as_posix())
    if peak is None:
        ctx.warn(
            f"{event.id}: normalize_source=true, но пик исходника измерить не "
            f"удалось — уровень оставлен как есть"
        )
        return 0.0
    # Запас общий для всех таких событий, поэтому баланс между ними не меняется.
    return -peak + ctx.timeline.mix.absolute_headroom_db


def final_envelope(
    event: Event,
    duck_intervals: list[tuple[float, float]],
    ctx: RenderContext,
) -> list[VolumePoint]:
    """Огибающая, которая реально уходит в фильтр volume."""
    points = build_envelope(event, duck_intervals, ctx.timeline.ducking)
    compensation = source_compensation_db(event, ctx)
    if abs(compensation) < 1e-9:
        return points
    return [VolumePoint(t=p.t, db=p.db + compensation) for p in points]


def is_static_envelope(points: list[VolumePoint]) -> bool:
    """True, если громкость постоянна — тогда не нужен eval=frame."""
    if len(points) <= 1:
        return True
    first = points[0].db
    return all(abs(p.db - first) <= 1e-6 for p in points)


# --------------------------------------------------------------------------- #
# Зацикливание
# --------------------------------------------------------------------------- #


def loop_copies(need: float, source_duration: float, crossfade: float) -> int:
    """Сколько копий файла нужно, чтобы покрыть ``need`` секунд.

    Каждый equal-power кроссфейд «съедает» ``crossfade`` секунд, поэтому
    полная длина N копий равна ``N*dur - (N-1)*crossfade``.
    """
    if source_duration <= EPS:
        raise RenderError(f"нулевая длительность исходника при need={need}")
    if need <= source_duration + EPS:
        return 1
    usable = source_duration - crossfade
    if usable <= EPS:
        raise RenderError(
            f"crossfade={crossfade} с не меньше длины файла {source_duration} с — "
            f"уменьшите loop.crossfade в сценарии"
        )
    return max(1, math.ceil((need - crossfade) / usable))


def pan_gains(pan: float) -> tuple[float, float]:
    """Баланс без подъёма выше единицы: панорама только ослабляет дальний канал.

    Так исключён риск клиппинга от самой панорамы (требование «не допускать
    клиппинга»), а на pan=0 фильтр полностью прозрачен.
    """
    if pan < 0:
        return 1.0, 1.0 + pan
    if pan > 0:
        return 1.0 - pan, 1.0
    return 1.0, 1.0


# --------------------------------------------------------------------------- #
# Подготовка событий
# --------------------------------------------------------------------------- #


def resolve_events(ctx: RenderContext) -> tuple[list[Event], list[tuple[Event, str]]]:
    """Определяет фактические окна событий и отбрасывает недоступные файлы."""
    timeline = ctx.timeline
    usable: list[Event] = []
    skipped: list[tuple[Event, str]] = []

    for event in timeline.events:
        probe = ctx.probe(event.file)
        key = event.file.as_posix()

        if not probe.exists:
            reason = "файл не найден"
            if event.required:
                raise RenderError(f"{event.id}: отсутствует обязательный файл {key}")
            skipped.append((event, reason))
            continue
        if probe.error or not probe.duration:
            reason = probe.error or "нулевая длительность"
            if event.required:
                raise RenderError(f"{event.id}: файл {key} не читается — {reason}")
            skipped.append((event, reason))
            continue

        event.source_duration = probe.duration
        if event.end is not None:
            event.resolved_end = min(event.end, timeline.total_duration)
        else:
            event.resolved_end = min(
                event.start + probe.duration, timeline.total_duration
            )

        if event.resolved_end - event.start <= EPS:
            skipped.append((event, "нулевое окно после обрезки по общей длительности"))
            continue

        usable.append(event)

    return usable, skipped


def voice_duck_intervals(events: list[Event], timeline: Timeline) -> list[tuple[float, float]]:
    """Интервалы, в которые звучит хоть один голос — основа для приглушения."""
    raw = [
        (event.start, event.resolved_end)
        for event in events
        if event.stem == "voices" and event.resolved_end is not None
    ]
    return merge_intervals(raw)


# --------------------------------------------------------------------------- #
# Построение filter_complex
# --------------------------------------------------------------------------- #


def build_event_chain(
    event: Event,
    input_indexes: list[int],
    ctx: RenderContext,
    duck_intervals: list[tuple[float, float]],
) -> tuple[list[str], str]:
    """Цепочка фильтров для одного события. Возвращает (фильтры, выходная метка)."""
    timeline = ctx.timeline
    output_cfg = timeline.output
    label = _sanitize_label(event.id)
    start, end = event.window()
    need = end - start
    total = timeline.total_duration
    fmt = (
        f"aformat=sample_fmts=fltp:sample_rates={output_cfg.sample_rate}"
        f":channel_layouts={output_cfg.channel_layout}"
    )

    filters: list[str] = []

    # 1. Нормализуем каждую копию исходника к рабочему формату.
    copies: list[str] = []
    for i, index in enumerate(input_indexes):
        copy_label = f"{label}_c{i}"
        filters.append(
            f"[{index}:a]aresample={output_cfg.sample_rate}:first_pts=0,{fmt},"
            f"asetpts=PTS-STARTPTS[{copy_label}]"
        )
        copies.append(copy_label)

    # 2. Бесшовное зацикливание equal-power кроссфейдом (никаких резких стыков).
    current = copies[0]
    for i in range(1, len(copies)):
        joined = f"{label}_x{i}"
        filters.append(
            f"[{current}][{copies[i]}]acrossfade=d={timeline.loop.crossfade:.6f}"
            f":c1={timeline.loop.curve}:c2={timeline.loop.curve}[{joined}]"
        )
        current = joined

    # 3. Приводим к точной длине окна (apad добивает тишиной, atrim обрезает).
    chain = [
        f"apad=whole_dur={need:.6f}",
        f"atrim=duration={need:.6f}",
        "asetpts=PTS-STARTPTS",
    ]

    # 4. Панорама.
    if abs(event.pan) > 1e-9:
        left, right = pan_gains(event.pan)
        chain.append(f"pan={output_cfg.channel_layout}|c0={left:.6f}*c0|c1={right:.6f}*c1")

    # 5. Автоматизация громкости (ключевые точки + приглушение + компенсация
    #    уровня исходника, если включён normalize_source).
    envelope = final_envelope(event, duck_intervals, ctx)
    if is_static_envelope(envelope):
        amplitude = db_to_amp(envelope[0].db) if envelope else 1.0
        if abs(amplitude - 1.0) > 1e-9:
            chain.append(f"volume={amplitude:.9f}")
    else:
        chain.append(f"asetnsamples=n={ENVELOPE_FRAME_SAMPLES}:p=0")
        chain.append(f"volume='{envelope_to_expr(envelope)}':eval=frame")

    # 6. Ставим на место во времени и добиваем до общей длительности.
    if start > EPS:
        chain.append(f"adelay={round(start * 1000)}:all=1")
    chain.extend(
        [
            f"apad=whole_dur={total:.6f}",
            f"atrim=duration={total:.6f}",
            "asetpts=PTS-STARTPTS",
        ]
    )

    out_label = f"{label}_out"
    filters.append(f"[{current}]" + ",".join(chain) + f"[{out_label}]")
    return filters, out_label


def build_stem_command(
    stem: str,
    events: list[Event],
    ctx: RenderContext,
    out_path: Path,
    duck_intervals: list[tuple[float, float]],
) -> list[str]:
    """Собирает полную команду ffmpeg для одного stem."""
    timeline = ctx.timeline
    output_cfg = timeline.output
    total = timeline.total_duration
    fmt = (
        f"aformat=sample_fmts=fltp:sample_rates={output_cfg.sample_rate}"
        f":channel_layouts={output_cfg.channel_layout}"
    )

    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    filters: list[str] = []
    mix_labels: list[str] = []
    input_count = 0

    for event in sorted(events, key=lambda e: (e.start, e.id)):
        start, end = event.window()
        need = end - start
        copies = (
            loop_copies(need, event.source_duration or 0.0, timeline.loop.crossfade)
            if event.loop
            else 1
        )
        indexes: list[int] = []
        for _ in range(copies):
            argv.extend(["-i", arg_path(event.file, ctx.project_root)])
            indexes.append(input_count)
            input_count += 1

        event_filters, label = build_event_chain(event, indexes, ctx, duck_intervals)
        filters.extend(event_filters)
        mix_labels.append(label)

    if len(mix_labels) == 1:
        filters.append(f"[{mix_labels[0]}]{fmt}[out]")
    else:
        joined = "".join(f"[{label}]" for label in mix_labels)
        # normalize=0: amix не должен делить сумму на число входов —
        # уровни уже выставлены сценарием.
        filters.append(
            f"{joined}amix=inputs={len(mix_labels)}:normalize=0:dropout_transition=0,"
            f"{fmt},atrim=duration={total:.6f},asetpts=PTS-STARTPTS[out]"
        )

    argv.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{total:.6f}",
            "-ar",
            str(output_cfg.sample_rate),
            "-ac",
            str(output_cfg.channels),
            "-c:a",
            output_cfg.pcm_codec,
            arg_path(out_path, ctx.project_root),
        ]
    )
    return argv


def build_silence_command(ctx: RenderContext, out_path: Path) -> list[str]:
    """Команда для пустой категории: корректный stem нужной длины из тишины."""
    output_cfg = ctx.timeline.output
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={output_cfg.sample_rate}:cl={output_cfg.channel_layout}",
        "-t",
        f"{ctx.timeline.total_duration:.6f}",
        "-ar",
        str(output_cfg.sample_rate),
        "-ac",
        str(output_cfg.channels),
        "-c:a",
        output_cfg.pcm_codec,
        arg_path(out_path, ctx.project_root),
    ]


# --------------------------------------------------------------------------- #
# Запуск ffmpeg
# --------------------------------------------------------------------------- #


def run_ffmpeg(ctx: RenderContext, stage: str, argv: list[str]) -> str:
    """Запускает ffmpeg, записывает команду в журнал и возвращает stderr."""
    ctx.commands.append(CommandRecord(stage=stage, argv=argv))
    if ctx.verbose:
        print(f"  [cmd]   {' '.join(_quote(a) for a in argv)}")

    try:
        proc = subprocess.run(
            argv,
            cwd=ctx.project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RenderError(f"{stage}: не удалось запустить ffmpeg — {exc}") from exc

    stderr = proc.stderr or ""
    if ctx.verbose and stderr.strip():
        for line in stderr.strip().splitlines():
            print(f"          {line}")

    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        raise RenderError(f"{stage}: ffmpeg завершился с кодом {proc.returncode}\n{tail}")
    return stderr


def measure_peak_db(ctx: RenderContext, path: Path, stage: str) -> float | None:
    """Максимальный сэмпловый пик файла в dBFS.

    Используется astats, а не volumedetect: volumedetect считает во внутреннем
    s16, поэтому для тишины показывает «-91 dB» вместо настоящего нуля и не
    различает уровни ниже 16-битного шага. astats работает во float.
    Возвращает None, если в файле ровная тишина.
    """
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        arg_path(path, ctx.project_root),
        "-af",
        "astats=measure_perchannel=none:measure_overall=Peak_level",
        "-f",
        "null",
        "-",
    ]
    stderr = run_ffmpeg(ctx, stage, argv)
    tail = stderr[stderr.rfind("Overall") :] if "Overall" in stderr else stderr
    match = re.search(r"Peak level dB:\s*(-?(?:inf|\d+(?:\.\d+)?))", tail)
    if not match:
        return None
    value = match.group(1)
    if "inf" in value:
        return None
    return round(float(value), 2)


def _parse_loudnorm_json(stderr: str) -> dict:
    """Достаёт JSON-блок, который loudnorm печатает в stderr."""
    matches = re.findall(r"\{[^{}]*\}", stderr, re.DOTALL)
    for block in reversed(matches):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "input_i" in data or "output_i" in data:
            return data
    raise RenderError(
        "не удалось прочитать измерения loudnorm из вывода ffmpeg "
        "(проверьте версию FFmpeg — нужен фильтр loudnorm)"
    )


def analyze_loudness(ctx: RenderContext, path: Path, stage: str) -> dict:
    """Первый проход loudnorm: измеряем громкость, ничего не записываем."""
    output_cfg = ctx.timeline.output
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        arg_path(path, ctx.project_root),
        "-af",
        (
            f"loudnorm=I={output_cfg.target_lufs}:TP={output_cfg.target_true_peak}"
            f":LRA={output_cfg.target_lra}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    return _parse_loudnorm_json(run_ffmpeg(ctx, stage, argv))


# --------------------------------------------------------------------------- #
# Stems и master
# --------------------------------------------------------------------------- #


def render_stems(
    ctx: RenderContext,
    usable: list[Event],
    skipped: list[tuple[Event, str]],
    output_dir: Path,
    duck_intervals: list[tuple[float, float]],
) -> list[StemResult]:
    """Рендерит по одному файлу на каждую категорию из timeline.stems."""
    results: list[StemResult] = []
    skipped_by_stem: dict[str, list[str]] = {}
    for event, _ in skipped:
        skipped_by_stem.setdefault(event.stem, []).append(event.id)

    for stem in ctx.timeline.stems:
        stem_events = [event for event in usable if event.stem == stem]
        out_path = output_dir / f"{stem}.wav"
        silent = not stem_events

        if silent:
            print(f"  [stem]  {stem}: событий нет — генерирую тишину "
                  f"{format_timecode(ctx.timeline.total_duration)}")
            argv = build_silence_command(ctx, out_path)
        else:
            print(f"  [stem]  {stem}: событий — {len(stem_events)}")
            argv = build_stem_command(stem, stem_events, ctx, out_path, duck_intervals)

        run_ffmpeg(ctx, f"stem:{stem}", argv)
        # probe_file принимает и относительный, и абсолютный путь:
        # project_root / absolute даёт сам absolute.
        probe = probe_file(out_path, ctx.project_root)
        peak = measure_peak_db(ctx, out_path, f"peak:{stem}")

        if peak is not None and peak > -0.1:
            ctx.warn(
                f"stem {stem}: пик {peak:.2f} dBFS — близко к клиппингу, "
                f"уменьшите уровни в scenario/timeline.json"
            )

        results.append(
            StemResult(
                name=stem,
                path=out_path,
                event_ids=[event.id for event in stem_events],
                skipped_event_ids=skipped_by_stem.get(stem, []),
                silent=silent,
                duration=probe.duration,
                peak_db=peak,
            )
        )
    return results


def build_premaster_command(
    ctx: RenderContext, stems: list[StemResult], out_path: Path
) -> list[str]:
    """Сумма всех stems во float32 — промежуточный микс без риска клиппинга."""
    output_cfg = ctx.timeline.output
    fmt = (
        f"aformat=sample_fmts=fltp:sample_rates={output_cfg.sample_rate}"
        f":channel_layouts={output_cfg.channel_layout}"
    )
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for stem in stems:
        argv.extend(["-i", arg_path(stem.path, ctx.project_root)])

    joined = "".join(f"[{i}:a]" for i in range(len(stems)))
    graph = (
        f"{joined}amix=inputs={len(stems)}:normalize=0:dropout_transition=0,{fmt},"
        f"atrim=duration={ctx.timeline.total_duration:.6f},asetpts=PTS-STARTPTS[out]"
    )
    argv.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-t",
            f"{ctx.timeline.total_duration:.6f}",
            "-ar",
            str(output_cfg.sample_rate),
            "-ac",
            str(output_cfg.channels),
            "-c:a",
            "pcm_f32le",
            arg_path(out_path, ctx.project_root),
        ]
    )
    return argv


def render_master(
    ctx: RenderContext,
    stems: list[StemResult],
    output_dir: Path,
    normalize: bool,
    keep_intermediate: bool,
) -> tuple[Path, LoudnessResult, float | None]:
    """Сводит stems в master.wav и приводит громкость к целевым значениям."""
    output_cfg = ctx.timeline.output
    master_path = output_dir / "master.wav"
    premaster_path = output_dir / "_premaster.wav"

    print("  [mix]   свожу stems в промежуточный микс (float32)")
    run_ffmpeg(ctx, "premaster", build_premaster_command(ctx, stems, premaster_path))

    loudness = LoudnessResult(applied=normalize, target_lufs=output_cfg.target_lufs)
    # Целимся ниже требуемого потолка на запас, иначе замер выходит на
    # несколько сотых dB выше -1.5 и формально нарушает требование.
    peak_ceiling = output_cfg.target_true_peak - output_cfg.true_peak_margin_db
    limit_amp = db_to_amp(peak_ceiling)

    if normalize:
        print("  [loud]  проход 1/2: измеряю громкость микса")
        measured = analyze_loudness(ctx, premaster_path, "loudnorm:analyze")
        loudness.measured_input = measured

        measured_i = float(measured["input_i"])
        measured_tp = float(measured["input_tp"])
        requested_gain = output_cfg.target_lufs - measured_i
        loudness.requested_gain_db = round(requested_gain, 2)
        effective_target = output_cfg.target_lufs

        # Прирост, при котором loudnorm ещё остаётся линейным: пик микса
        # не должен перелезть потолок True Peak.
        # 0.5 dB отступа обязательны: loudnorm остаётся линейным по условию
        # measured_TP + прирост <= target_TP, и ровно на границе сравнение
        # проваливается на округлении float — режим молча становится dynamic.
        max_linear_gain = peak_ceiling - measured_tp - 0.5
        loudness.max_linear_gain_db = round(max_linear_gain, 2)

        # loudnorm уходит в динамический режим по двум независимым причинам:
        # прирост не влезает в потолок True Peak ИЛИ измеренный LRA больше
        # целевого (линейно сузить диапазон нельзя). Вторую причину снимаем,
        # подняв целевой LRA выше измеренного: сужать диапазон нам и не нужно.
        target_lra = output_cfg.target_lra
        measured_lra = float(measured["input_lra"])
        if output_cfg.preserve_dynamics and measured_lra >= target_lra:
            target_lra = min(50.0, measured_lra + 2.0)

        if output_cfg.preserve_dynamics and requested_gain > max_linear_gain:
            effective_target = measured_i + max_linear_gain
            loudness.gain_capped = True
            loudness.cap_reason = "preserve_dynamics"
            ctx.warn(
                f"preserve_dynamics=true: чтобы loudnorm остался линейным, "
                f"прирост ограничен {max_linear_gain:.1f} dB (crest микса "
                f"{measured_tp - measured_i:.1f} dB) → фактическая громкость "
                f"около {effective_target:.1f} LUFS вместо "
                f"{output_cfg.target_lufs:.1f} LUFS. Уровни сценария при этом "
                f"сохраняются точно."
            )
        elif requested_gain > output_cfg.max_normalize_gain_db:
            effective_target = measured_i + output_cfg.max_normalize_gain_db
            loudness.gain_capped = True
            loudness.cap_reason = "max_normalize_gain_db"
            ctx.warn(
                f"микс измерен как {measured_i:.1f} LUFS: до цели "
                f"{output_cfg.target_lufs:.1f} LUFS не хватает "
                f"{requested_gain:.1f} dB, прирост ограничен "
                f"{output_cfg.max_normalize_gain_db:.1f} dB → фактическая цель "
                f"{effective_target:.1f} LUFS. Это ожидаемо, пока в миксе нет "
                f"голосов; после добавления реплик ограничение снимется само. "
                f"Порог меняется в scenario/timeline.json "
                f"(output.max_normalize_gain_db)."
            )

        loudness.effective_target_lufs = round(effective_target, 2)
        loudness.applied_gain_db = round(effective_target - measured_i, 2)

        print(
            f"  [loud]  проход 2/2: {measured_i:.1f} LUFS → "
            f"{effective_target:.1f} LUFS, True Peak ≤ "
            f"{output_cfg.target_true_peak:.1f} dB "
            f"(целимся в {peak_ceiling:.1f} dB с запасом)"
        )
        # linear=true — один постоянный коэффициент на всю дорожку: динамика
        # сценария сохраняется, результат воспроизводим.
        # aresample после loudnorm обязателен: фильтр работает на 192 кГц.
        loudnorm_filter = (
            f"loudnorm=I={effective_target:.6f}:TP={peak_ceiling:.6f}"
            f":LRA={target_lra:.6f}"
            f":measured_I={float(measured['input_i']):.6f}"
            f":measured_LRA={float(measured['input_lra']):.6f}"
            f":measured_TP={float(measured['input_tp']):.6f}"
            f":measured_thresh={float(measured['input_thresh']):.6f}"
            f":offset={float(measured['target_offset']):.6f}"
            f":linear=true:print_format=json"
        )
        graph = (
            f"{loudnorm_filter},aresample={output_cfg.sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts={output_cfg.channel_layout},"
            f"alimiter=limit={limit_amp:.6f}:level=disabled"
        )
    else:
        print(
            f"  [loud]  нормализация отключена (--no-normalize): только лимитер "
            f"на {peak_ceiling:.1f} dB (потолок True Peak "
            f"{output_cfg.target_true_peak:.1f} dB)"
        )
        graph = f"alimiter=limit={limit_amp:.6f}:level=disabled"

    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        arg_path(premaster_path, ctx.project_root),
        "-af",
        graph,
        "-t",
        f"{ctx.timeline.total_duration:.6f}",
        "-ar",
        str(output_cfg.sample_rate),
        "-ac",
        str(output_cfg.channels),
        "-c:a",
        output_cfg.pcm_codec,
        arg_path(master_path, ctx.project_root),
    ]
    master_stderr = run_ffmpeg(ctx, "master", argv)

    if normalize:
        # loudnorm может молча проигнорировать linear=true. Единственный
        # надёжный способ узнать, что он сделал, — прочитать его собственный
        # отчёт: тихое переключение на dynamic переформовывает динамику.
        try:
            applied = _parse_loudnorm_json(master_stderr)
        except RenderError:
            applied = {}
        loudness.normalization_type = applied.get("normalization_type")
        if loudness.normalization_type == "dynamic":
            ctx.warn(
                "loudnorm перешёл в динамический режим: прирост неодинаков по "
                "дорожке, тихие места поднимаются сильнее громких, и заданная "
                "в сценарии динамика частично переформована. Чтобы этого не "
                "было, поставьте output.preserve_dynamics=true — тогда "
                "громкость окажется ниже цели, но уровни сохранятся точно."
            )

    print("  [loud]  контрольное измерение master.wav")
    loudness.measured_output = analyze_loudness(ctx, master_path, "loudnorm:verify")
    master_peak = measure_peak_db(ctx, master_path, "peak:master")

    achieved_i = float(loudness.measured_output["input_i"])
    achieved_tp = float(loudness.measured_output["input_tp"])
    if achieved_tp > output_cfg.target_true_peak:
        ctx.warn(
            f"master: True Peak {achieved_tp:.2f} dB превышает потолок "
            f"{output_cfg.target_true_peak:.1f} dB — увеличьте "
            f"output.true_peak_margin_db в scenario/timeline.json"
        )
    if master_peak is not None and master_peak > -0.01:
        ctx.warn(f"master: сэмпловый пик {master_peak:.2f} dBFS — возможен клиппинг")
    if normalize and not loudness.gain_capped and abs(achieved_i - output_cfg.target_lufs) > 1.0:
        ctx.warn(
            f"master: измерено {achieved_i:.1f} LUFS вместо целевых "
            f"{output_cfg.target_lufs:.1f} LUFS"
        )

    if not keep_intermediate:
        premaster_path.unlink(missing_ok=True)
    else:
        print(f"  [mix]   промежуточный микс сохранён: {premaster_path.name}")

    return master_path, loudness, master_peak


def write_command_log(ctx: RenderContext, path: Path) -> None:
    """Сохраняет все выполненные команды ffmpeg в читаемом виде."""
    lines = [
        "# Команды FFmpeg, выполненные при последней сборке.",
        "# Рабочий каталог для всех команд — корень проекта.",
        f"# Всего команд: {len(ctx.commands)}",
        "",
    ]
    for i, record in enumerate(ctx.commands, start=1):
        lines.append(f"## [{i}] {record.stage}")
        lines.append(record.as_text())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def envelope_summary(
    event: Event, duck_intervals: list[tuple[float, float]], ducking: DuckingConfig
) -> list[dict]:
    """Огибающая события в удобном для отчёта виде (абсолютное время)."""
    start, _ = event.window()
    return [
        {"t": format_timecode(start + point.t), "db": round(point.db, 2)}
        for point in build_envelope(event, duck_intervals, ducking)
    ]


def measure_source_peaks(ctx: RenderContext, events: list[Event]) -> None:
    """Замеряет пик каждого исходника — по одному проходу на уникальный файл."""
    for event in events:
        key = event.file.as_posix()
        if key in ctx.source_peaks:
            continue
        ctx.source_peaks[key] = measure_peak_db(
            ctx, event.file, f"peak:src:{event.id}"
        )


def expected_level(
    event: Event,
    ctx: RenderContext,
    duck_intervals: list[tuple[float, float]],
) -> dict:
    """Какой уровень событие получит в миксе.

    Значения dB в сценарии — это ОТНОСИТЕЛЬНОЕ ослабление, а не абсолютная
    цель. У исходников уровни различаются очень сильно, поэтому одинаковый
    gain даёт совершенно разную слышимость. Этот расчёт делает разницу видимой,
    не требуя отдельного прохода на каждое событие.
    """
    source_peak = ctx.source_peaks.get(event.file.as_posix())
    scenario = build_envelope(event, duck_intervals, ctx.timeline.ducking)
    loudest = max((point.db for point in scenario), default=0.0)
    compensation = source_compensation_db(event, ctx)
    return {
        "source_peak_dbfs": source_peak,
        "normalize_source": event.normalize_source,
        "source_compensation_db": round(compensation, 2),
        "max_envelope_db": round(loudest, 2),
        "expected_peak_dbfs": (
            round(source_peak + compensation + loudest, 2)
            if source_peak is not None
            else None
        ),
    }


# Ниже этого расчётного пика событие практически не слышно в миксе.
INAUDIBLE_PEAK_DBFS = -55.0
# Выше этого — событие рискует упереться в потолок.
HOT_PEAK_DBFS = -1.0


def _warn_on_level_imbalance(
    ctx: RenderContext,
    events: list[Event],
    duck_intervals: list[tuple[float, float]],
) -> None:
    """Предупреждает, если событие окажется неслышимым или слишком горячим."""
    for event in sorted(events, key=lambda e: (e.start, e.id)):
        level = expected_level(event, ctx, duck_intervals)
        peak = level["expected_peak_dbfs"]
        if peak is None:
            continue
        if peak > HOT_PEAK_DBFS:
            ctx.warn(
                f"{event.id}: расчётный пик {peak:+.1f} dBFS — уменьшите gain_db "
                f"или volume_points в scenario/timeline.json"
            )
        elif peak < INAUDIBLE_PEAK_DBFS:
            ctx.warn(
                f"{event.id}: расчётный пик {peak:.1f} dBFS — событие почти не "
                f"будет слышно. Исходник {event.file.name} сам по себе тихий "
                f"(пик {level['source_peak_dbfs']:.1f} dBFS), а сценарий "
                f"добавляет ещё {level['max_envelope_db']:+.1f} dB"
            )


def render_all(
    ctx: RenderContext,
    output_dir: Path,
    normalize: bool = True,
    keep_intermediate: bool = False,
) -> dict:
    """Полный прогон: stems -> master -> данные для render-report.json."""
    usable, skipped = resolve_events(ctx)

    for event, reason in skipped:
        ctx.warn(
            f"{event.id} ({event.stem}): пропущено — {reason} "
            f"[{event.file.as_posix()}]"
        )

    duck_intervals = voice_duck_intervals(usable, ctx.timeline)
    if duck_intervals:
        windows = ", ".join(
            f"{format_timecode(a)}-{format_timecode(b)}" for a, b in duck_intervals
        )
        print(f"  [duck]  интервалы голосов: {windows}")
    else:
        print("  [duck]  голосов в миксе нет — приглушение не применяется")

    measure_source_peaks(ctx, usable)
    _warn_on_level_imbalance(ctx, usable, duck_intervals)

    stems = render_stems(ctx, usable, skipped, output_dir, duck_intervals)
    master_path, loudness, master_peak = render_master(
        ctx, stems, output_dir, normalize, keep_intermediate
    )

    master_probe = probe_file(master_path, ctx.project_root)

    return {
        "stems": stems,
        "master": {
            "path": master_path,
            "duration": master_probe.duration,
            "sample_rate": master_probe.sample_rate,
            "channels": master_probe.channels,
            "codec": master_probe.codec,
            "peak_db": master_peak,
        },
        "loudness": loudness,
        "usable_events": usable,
        "skipped_events": skipped,
        "duck_intervals": duck_intervals,
    }
