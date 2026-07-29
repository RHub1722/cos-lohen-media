"""Бенчмарк скорости нарезки аудио через FFmpeg.

Запуск::

    python benchmark/bench.py
    python benchmark/bench.py --label home-pc
    python benchmark/bench.py --quick
    python benchmark/bench.py --compare

Что меряется и зачем. Основной вопрос — сколько стоит нарезка аудиофайла и
из чего это время складывается. Замеры разбиты на три группы:

* «запуск»    — сколько занимает сам старт процесса ffmpeg, без работы;
* «нарезка»   — один срез тремя способами и партия из 20 срезов тремя
                стратегиями (по одному вызову на срез, все срезы одним
                вызовом, вызовы параллельно по числу ядер);
* «конвейер»  — стадии, из которых состоит настоящая сборка проекта:
                замер тишины по краям, замер пика, volume-автоматизация с
                eval=frame, микс через amix и первый проход loudnorm.

Бенчмарк не зависит от ``assets/``: исходники генерируются самим ffmpeg из
lavfi с фиксированным сидом, поэтому на любой машине материал одинаковый и
числа сравнимы. Зависимостей, кроме стандартной библиотеки и ffmpeg в PATH,
нет — как и у всего остального проекта.

Время выполнения не может быть детерминированным, поэтому каждый случай
прогоняется несколько раз, а для сравнения машин берётся МИНИМУМ: он меньше
всего искажён посторонней нагрузкой на машину. Медиана приводится рядом как
проверка стабильности замера.

Результаты пишутся в ``benchmark/results/<label>.json`` (для сравнения
машин) и ``benchmark/results/<label>.md`` (для чтения глазами).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
WORK_DIR = BENCH_ROOT / ".work"
SOURCE_DIR = WORK_DIR / "sources"
SLICE_DIR = WORK_DIR / "slices"
RESULTS_DIR = BENCH_ROOT / "results"

BENCHMARK_VERSION = 1

# Формат совпадает с output-секцией scenario/timeline.json: бенчмарк должен
# мерить ту же работу, что делает настоящая сборка.
SAMPLE_RATE = 48000
CHANNELS = 2
CHANNEL_LAYOUT = "stereo"
PCM_CODEC = "pcm_s24le"

# Длинный исходник равен total_duration сценария (01:43), короткий — типичный
# one-shot вроде удара или щелчка.
LONG_DURATION = 103.0
SHORT_DURATION = 3.0
NOISE_SEED = 20260730

# Партия срезов: 20 штук по 1.5 с, разнесённых по всему длинному файлу.
SLICE_COUNT = 20
SLICE_DURATION = 1.5
SLICE_STEP = 5.0

# Столько копий короткого файла микшируется в случае mix_amix.
MIX_INPUTS = 8

# Тот же размер блока, что в src/renderer.py: от него зависит, сколько раз
# фильтр volume пересчитает выражение.
ENVELOPE_FRAME_SAMPLES = 128

# Кусочно-линейная огибающая в том же виде, в каком её генерирует
# renderer.envelope_to_expr: вложенные if(lt(t,...)).
ENVELOPE_EXPR = (
    "if(lt(t,2.000000),(0.100000+0.200000*(t-0.000000)),"
    "if(lt(t,90.000000),0.500000,"
    "if(lt(t,95.000000),(0.500000-0.060000*(t-90.000000)),0.200000)))"
)

AFORMAT = (
    f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}"
    f":channel_layouts={CHANNEL_LAYOUT}"
)


class BenchError(Exception):
    """Бенчмарк выполнить невозможно."""


def configure_console() -> None:
    """Переводит вывод в UTF-8 — иначе русский текст падает в консоли Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # перенаправленный или закрытый поток
            pass


# --------------------------------------------------------------------------- #
# Запуск ffmpeg
# --------------------------------------------------------------------------- #


def run(argv: list[str]) -> None:
    """Выполняет ffmpeg и молчит при успехе. Вывод не разбирается."""
    try:
        proc = subprocess.run(
            argv,
            cwd=BENCH_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise BenchError(f"не удалось запустить ffmpeg — {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise BenchError(
            f"ffmpeg завершился с кодом {proc.returncode}\n"
            f"команда: {' '.join(argv)}\n{tail}"
        )


def ffmpeg_version() -> str:
    """Первая строка ``ffmpeg -version`` — попадает в отчёт."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    first = (proc.stdout or "").strip().splitlines()
    return first[0] if first else "неизвестно"


def require_ffmpeg() -> str:
    if shutil.which("ffmpeg") is None:
        raise BenchError(
            "ffmpeg не найден в PATH. Установите его "
            "(Windows: winget install Gyan.FFmpeg) и откройте новое окно консоли."
        )
    return ffmpeg_version()


def rel(path: Path) -> str:
    """Путь для командной строки — относительный от benchmark/ (cwd ffmpeg)."""
    try:
        return path.resolve().relative_to(BENCH_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- #
# Исходники
# --------------------------------------------------------------------------- #


def source_path(name: str) -> Path:
    return SOURCE_DIR / f"{name}.wav"


def generate_source(path: Path, duration: float) -> None:
    """Генерирует детерминированный исходник: тон + розовый шум с фиксированным сидом.

    Ни один файл из ``assets/`` не используется, поэтому бенчмарк работает на
    свежем клоне и даёт сравнимые числа на разных машинах.
    """
    graph = (
        f"sine=frequency=220:sample_rate={SAMPLE_RATE}:duration={duration:.6f}[tone];"
        f"anoisesrc=color=pink:sample_rate={SAMPLE_RATE}:seed={NOISE_SEED}"
        f":duration={duration:.6f}[noise];"
        f"[tone][noise]amix=inputs=2:normalize=0,volume=0.35,"
        f"pan={CHANNEL_LAYOUT}|c0=c0|c1=c0,{AFORMAT}[out]"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-t",
            f"{duration:.6f}",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-c:a",
            PCM_CODEC,
            rel(path),
        ]
    )


def prepare_sources(regen: bool) -> dict[str, bool]:
    """Создаёт исходники, если их ещё нет. Возвращает, что было переиспользовано."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    reused: dict[str, bool] = {}
    for name, duration in (("long", LONG_DURATION), ("short", SHORT_DURATION)):
        path = source_path(name)
        if regen or not path.exists() or path.stat().st_size == 0:
            print(f"  генерирую {rel(path)} ({duration:g} с)")
            generate_source(path, duration)
            reused[name] = False
        else:
            print(f"  переиспользую {rel(path)} ({path.stat().st_size / 1e6:.1f} МБ)")
            reused[name] = True
    return reused


# --------------------------------------------------------------------------- #
# Команды отдельных случаев
# --------------------------------------------------------------------------- #


def slice_offsets() -> list[float]:
    return [i * SLICE_STEP for i in range(SLICE_COUNT)]


def cmd_startup() -> list[str]:
    """Базовая стоимость старта процесса: ffmpeg ничего не обрабатывает."""
    return ["ffmpeg", "-hide_banner", "-version"]


def cmd_slice_copy(start: float, out: Path) -> list[str]:
    """Срез без перекодирования: input-seek плюс ``-c copy``."""
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{start:.6f}",
        "-t", f"{SLICE_DURATION:.6f}",
        "-i", rel(source_path("long")),
        "-c", "copy",
        rel(out),
    ]


def cmd_slice_encode(start: float, out: Path) -> list[str]:
    """Срез с перекодированием в целевой PCM — декод плюс энкод."""
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{start:.6f}",
        "-t", f"{SLICE_DURATION:.6f}",
        "-i", rel(source_path("long")),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-c:a", PCM_CODEC,
        rel(out),
    ]


def cmd_slice_atrim(start: float, out: Path) -> list[str]:
    """Срез фильтром ``atrim`` — так режет настоящая сборка (src/renderer.py)."""
    graph = (
        f"[0:a]atrim=start={start:.6f}:duration={SLICE_DURATION:.6f},"
        f"asetpts=PTS-STARTPTS,{AFORMAT}[out]"
    )
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", rel(source_path("long")),
        "-filter_complex", graph,
        "-map", "[out]",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-c:a", PCM_CODEC,
        rel(out),
    ]


def cmd_slice_batch_single() -> list[str]:
    """Все 20 срезов одним вызовом: asplit плюс по atrim на каждый выход.

    Сравнение с 20 отдельными вызовами показывает, сколько в партии занимает
    не обработка, а повторный запуск процесса.
    """
    offsets = slice_offsets()
    labels = "".join(f"[s{i}]" for i in range(len(offsets)))
    parts = [f"[0:a]asplit={len(offsets)}{labels}"]
    for i, start in enumerate(offsets):
        parts.append(
            f"[s{i}]atrim=start={start:.6f}:duration={SLICE_DURATION:.6f},"
            f"asetpts=PTS-STARTPTS,{AFORMAT}[o{i}]"
        )

    argv = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", rel(source_path("long")),
        "-filter_complex", ";".join(parts),
    ]
    for i in range(len(offsets)):
        argv.extend(
            [
                "-map", f"[o{i}]",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "-c:a", PCM_CODEC,
                rel(SLICE_DIR / f"single_{i:02d}.wav"),
            ]
        )
    return argv


def cmd_silence_edge(reverse: bool) -> list[str]:
    """Замер тишины с одного края — стадия silence из src/renderer.py."""
    threshold = "-60dB" if reverse else "-50dB"
    chain = (
        f"silenceremove=start_periods=1:start_duration=0"
        f":start_threshold={threshold}:detection=peak"
    )
    if reverse:
        chain = f"areverse,{chain},areverse"
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", rel(source_path("short")),
        "-af", chain,
        "-f", "null", "-",
    ]


def cmd_peak_astats() -> list[str]:
    """Замер пика через astats — стадия peak из src/renderer.py."""
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", rel(source_path("short")),
        "-af", "astats=measure_perchannel=none:measure_overall=Peak_level",
        "-f", "null", "-",
    ]


def cmd_volume_envelope() -> list[str]:
    """Автоматизация громкости с eval=frame на всей длине.

    Самая чувствительная к процессору стадия: выражение пересчитывается
    каждые 128 сэмплов, то есть около 38 000 раз на 103 секундах.
    """
    graph = (
        f"asetnsamples=n={ENVELOPE_FRAME_SAMPLES}:p=0,"
        f"volume='{ENVELOPE_EXPR}':eval=frame,{AFORMAT}"
    )
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", rel(source_path("long")),
        "-af", graph,
        "-f", "null", "-",
    ]


def cmd_mix_amix() -> list[str]:
    """Микс 8 копий короткого файла с задержкой и огибающей — как рендер стема."""
    total = SHORT_DURATION + MIX_INPUTS * 0.4
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for _ in range(MIX_INPUTS):
        argv.extend(["-i", rel(source_path("short"))])

    parts: list[str] = []
    for i in range(MIX_INPUTS):
        delay = round(i * 400)
        parts.append(
            f"[{i}:a]{AFORMAT},asetpts=PTS-STARTPTS,"
            f"asetnsamples=n={ENVELOPE_FRAME_SAMPLES}:p=0,"
            f"volume='{ENVELOPE_EXPR}':eval=frame,"
            f"adelay={delay}:all=1,"
            f"apad=whole_dur={total:.6f},atrim=duration={total:.6f},"
            f"asetpts=PTS-STARTPTS[e{i}]"
        )
    joined = "".join(f"[e{i}]" for i in range(MIX_INPUTS))
    parts.append(
        f"{joined}amix=inputs={MIX_INPUTS}:normalize=0:dropout_transition=0,"
        f"{AFORMAT}[out]"
    )

    argv.extend(
        [
            "-filter_complex", ";".join(parts),
            "-map", "[out]",
            "-t", f"{total:.6f}",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", PCM_CODEC,
            rel(SLICE_DIR / "mix.wav"),
        ]
    )
    return argv


def cmd_loudnorm_analyze() -> list[str]:
    """Первый проход loudnorm — в сборке это самая дорогая одиночная стадия."""
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", rel(source_path("long")),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]


# --------------------------------------------------------------------------- #
# Сценарии-обёртки
# --------------------------------------------------------------------------- #


def case_slice_batch_separate() -> None:
    """20 срезов — 20 отдельных вызовов ffmpeg, последовательно."""
    for i, start in enumerate(slice_offsets()):
        run(cmd_slice_encode(start, SLICE_DIR / f"sep_{i:02d}.wav"))


def case_slice_batch_parallel(workers: int) -> None:
    """20 срезов — те же 20 вызовов, но параллельно по числу ядер.

    Потоки, а не процессы: ``subprocess.run`` отпускает GIL на всё время
    ожидания ffmpeg, поэтому GIL здесь ничему не мешает.
    """
    jobs = [
        cmd_slice_encode(start, SLICE_DIR / f"par_{i:02d}.wav")
        for i, start in enumerate(slice_offsets())
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))


def case_silence_edges() -> None:
    """Оба края, как в сборке: два отдельных вызова на один файл."""
    run(cmd_silence_edge(reverse=False))
    run(cmd_silence_edge(reverse=True))


# --------------------------------------------------------------------------- #
# Замер
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    name: str
    title: str
    group: str
    calls: int  # вызовов ffmpeg внутри одного повтора
    fn: object
    note: str = ""


@dataclass
class CaseResult:
    case: Case
    samples_ms: list[float] = field(default_factory=list)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def spread_pct(self) -> float:
        """Разброс медианы относительно минимума — насколько замер стабилен."""
        return (self.median_ms / self.min_ms - 1.0) * 100.0 if self.min_ms else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.case.name,
            "title": self.case.title,
            "group": self.case.group,
            "ffmpeg_calls": self.case.calls,
            "repeats": len(self.samples_ms),
            "min_ms": round(self.min_ms, 1),
            "median_ms": round(self.median_ms, 1),
            "mean_ms": round(statistics.fmean(self.samples_ms), 1),
            "spread_pct": round(self.spread_pct, 1),
            "per_call_ms": round(self.min_ms / self.case.calls, 1),
            "note": self.case.note or None,
        }


def measure(case: Case, repeats: int, warmup: int) -> CaseResult:
    """Прогревочные прогоны в результат не идут: они греют кэш файловой системы."""
    for _ in range(warmup):
        case.fn()
    result = CaseResult(case=case)
    for _ in range(repeats):
        started = time.perf_counter()
        case.fn()
        result.samples_ms.append((time.perf_counter() - started) * 1000.0)
    return result


def build_cases(workers: int) -> list[Case]:
    out_single = SLICE_DIR / "one.wav"
    return [
        Case(
            name="startup",
            title="Пустой запуск ffmpeg (-version)",
            group="запуск",
            calls=1,
            fn=lambda: run(cmd_startup()),
            note="накладные расходы на создание процесса, работы нет",
        ),
        Case(
            name="slice_copy",
            title="1 срез, без перекодирования (-c copy)",
            group="нарезка",
            calls=1,
            fn=lambda: run(cmd_slice_copy(10.0, out_single)),
        ),
        Case(
            name="slice_encode",
            title="1 срез, с перекодированием в pcm_s24le",
            group="нарезка",
            calls=1,
            fn=lambda: run(cmd_slice_encode(10.0, out_single)),
        ),
        Case(
            name="slice_atrim",
            title="1 срез фильтром atrim (способ сборки)",
            group="нарезка",
            calls=1,
            fn=lambda: run(cmd_slice_atrim(10.0, out_single)),
        ),
        Case(
            name="slice_batch_separate",
            title=f"{SLICE_COUNT} срезов — {SLICE_COUNT} вызовов, последовательно",
            group="нарезка",
            calls=SLICE_COUNT,
            fn=case_slice_batch_separate,
        ),
        Case(
            name="slice_batch_single",
            title=f"{SLICE_COUNT} срезов — один вызов (asplit + atrim)",
            group="нарезка",
            calls=1,
            fn=lambda: run(cmd_slice_batch_single()),
        ),
        Case(
            name="slice_batch_parallel",
            title=f"{SLICE_COUNT} срезов — {SLICE_COUNT} вызовов в {workers} потоках",
            group="нарезка",
            calls=SLICE_COUNT,
            fn=lambda: case_slice_batch_parallel(workers),
        ),
        Case(
            name="silence_edges",
            title="Замер тишины по двум краям (2 вызова)",
            group="конвейер",
            calls=2,
            fn=case_silence_edges,
        ),
        Case(
            name="peak_astats",
            title="Замер пика через astats",
            group="конвейер",
            calls=1,
            fn=lambda: run(cmd_peak_astats()),
        ),
        Case(
            name="volume_envelope",
            title=f"volume eval=frame на {LONG_DURATION:g} с",
            group="конвейер",
            calls=1,
            fn=lambda: run(cmd_volume_envelope()),
            note=f"выражение считается каждые {ENVELOPE_FRAME_SAMPLES} сэмплов",
        ),
        Case(
            name="mix_amix",
            title=f"Микс {MIX_INPUTS} событий с огибающими (amix)",
            group="конвейер",
            calls=1,
            fn=lambda: run(cmd_mix_amix()),
        ),
        Case(
            name="loudnorm_analyze",
            title=f"loudnorm, проход измерения, {LONG_DURATION:g} с",
            group="конвейер",
            calls=1,
            fn=lambda: run(cmd_loudnorm_analyze()),
            note="внутри ресемпл в 192 кГц и последовательный гейтинг EBU R128",
        ),
    ]


# --------------------------------------------------------------------------- #
# Окружение и выводы
# --------------------------------------------------------------------------- #


def cpu_name() -> str:
    """Модель процессора. platform.processor() на Windows часто пустой."""
    for value in (platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")):
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return "неизвестно"


def collect_environment(ffmpeg: str) -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "ffmpeg": ffmpeg,
    }


def derive_insights(results: dict[str, CaseResult], workers: int) -> dict:
    """Считает выводы из чисел, чтобы их не приходилось выводить в уме."""
    insights: dict[str, object] = {}

    startup = results["startup"].min_ms
    encode = results["slice_encode"].min_ms
    insights["startup_ms"] = round(startup, 1)
    insights["startup_share_of_slice_pct"] = round(startup / encode * 100.0, 1)

    separate = results["slice_batch_separate"].min_ms
    single = results["slice_batch_single"].min_ms
    parallel = results["slice_batch_parallel"].min_ms
    insights["batch_separate_ms"] = round(separate, 1)
    insights["batch_single_speedup"] = round(separate / single, 2)
    insights["batch_parallel_speedup"] = round(separate / parallel, 2)
    insights["parallel_workers"] = workers
    insights["parallel_efficiency_pct"] = round(
        separate / parallel / workers * 100.0, 1
    )

    # Во сколько раз нарезка идёт быстрее реального времени материала.
    sliced_seconds = SLICE_COUNT * SLICE_DURATION
    insights["batch_realtime_factor_separate"] = round(
        sliced_seconds / (separate / 1000.0), 1
    )
    insights["batch_realtime_factor_parallel"] = round(
        sliced_seconds / (parallel / 1000.0), 1
    )

    loudnorm = results["loudnorm_analyze"].min_ms
    insights["loudnorm_realtime_factor"] = round(
        LONG_DURATION / (loudnorm / 1000.0), 1
    )
    insights["envelope_realtime_factor"] = round(
        LONG_DURATION / (results["volume_envelope"].min_ms / 1000.0), 1
    )
    return insights


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


GROUP_TITLES = {
    "запуск": "Накладные расходы",
    "нарезка": "Нарезка аудио",
    "конвейер": "Стадии конвейера сборки",
}


def write_markdown(path: Path, payload: dict) -> None:
    env = payload["environment"]
    cfg = payload["config"]
    ins = payload["insights"]

    lines = [
        f"# Бенчмарк нарезки аудио — {payload['label']}",
        "",
        f"Снято: {payload['generated_at']}  ",
        f"Версия бенчмарка: {payload['benchmark_version']}",
        "",
        "## Машина",
        "",
        "| | |",
        "|---|---|",
        f"| Имя | `{env['hostname']}` |",
        f"| ОС | {env['platform']} |",
        f"| Процессор | {env['cpu']} |",
        f"| Логических ядер | {env['cpu_count']} |",
        f"| Python | {env['python']} |",
        f"| FFmpeg | {env['ffmpeg']} |",
        "",
        f"Повторов на случай: {cfg['repeats']} (плюс {cfg['warmup']} прогревочных). "
        f"Для сравнения машин берите столбец «мин».",
        "",
    ]

    for group in ("запуск", "нарезка", "конвейер"):
        rows = [c for c in payload["cases"] if c["group"] == group]
        if not rows:
            continue
        lines.extend(
            [
                f"## {GROUP_TITLES[group]}",
                "",
                "| Случай | Вызовов | Мин, мс | Медиана, мс | Разброс | На вызов, мс |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['title']} | {row['ffmpeg_calls']} | **{row['min_ms']:.1f}** | "
                f"{row['median_ms']:.1f} | +{row['spread_pct']:.1f}% | "
                f"{row['per_call_ms']:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Выводы",
            "",
            f"* Пустой запуск ffmpeg — **{ins['startup_ms']:.0f} мс**, это "
            f"**{ins['startup_share_of_slice_pct']:.0f}%** времени одного среза "
            f"с перекодированием. Чем короче срез, тем большую долю занимает "
            f"не обработка, а сам старт процесса.",
            f"* Партия из {cfg['slice_count']} срезов одним вызовом ffmpeg быстрее "
            f"{cfg['slice_count']} отдельных вызовов в "
            f"**{ins['batch_single_speedup']:.1f}×**.",
            f"* Те же {cfg['slice_count']} вызовов в {ins['parallel_workers']} потоках "
            f"быстрее последовательных в **{ins['batch_parallel_speedup']:.1f}×** "
            f"(эффективность масштабирования "
            f"{ins['parallel_efficiency_pct']:.0f}%).",
            f"* Скорость относительно реального времени: последовательно "
            f"**×{ins['batch_realtime_factor_separate']:.0f}**, параллельно "
            f"**×{ins['batch_realtime_factor_parallel']:.0f}**.",
            f"* `loudnorm` обрабатывает материал со скоростью "
            f"**×{ins['loudnorm_realtime_factor']:.0f}** к реальному времени — "
            f"самая медленная стадия конвейера. `volume` с `eval=frame` — "
            f"**×{ins['envelope_realtime_factor']:.0f}**.",
            "",
            "## Про GPU",
            "",
            "Варианта «на GPU» в бенчмарке нет намеренно: в FFmpeg отсутствуют "
            "аудиофильтры с аппаратным ускорением. Проверить на своей сборке:",
            "",
            "```bash",
            "ffmpeg -hide_banner -filters | grep -iE \"cuda|opencl|vulkan|qsv\" | "
            "grep -E \"A+->A+\"",
            "```",
            "",
            "Пустой вывод означает, что ни один аудиофильтр GPU не использует. "
            "Все методы из `ffmpeg -hwaccels` относятся только к видео.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Сравнение машин
# --------------------------------------------------------------------------- #


def compare(paths: list[Path]) -> int:
    """Печатает результаты всех машин рядом. База сравнения — первый файл."""
    runs = []
    for path in sorted(paths):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [warn] {path.name} пропущен — {exc}")
    if not runs:
        print("В benchmark/results/ нет пригодных JSON-файлов.")
        return 1

    labels = [r["label"] for r in runs]
    print("=" * 72)
    print("Сравнение машин (минимум, мс; в скобках — отношение к первой)")
    print("=" * 72)
    for run_data in runs:
        env = run_data["environment"]
        print(
            f"  {run_data['label']:<16} {env['cpu_count']} ядер, "
            f"{env['cpu'][:44]}"
        )
    print()

    width = max(len(label) for label in labels) + 10
    header = "Случай".ljust(38) + "".join(label.ljust(width) for label in labels)
    print(header)
    print("-" * len(header))

    base_cases = {c["name"]: c for c in runs[0]["cases"]}
    for name, base in base_cases.items():
        row = base["title"][:36].ljust(38)
        for i, run_data in enumerate(runs):
            case = next((c for c in run_data["cases"] if c["name"] == name), None)
            if case is None:
                row += "—".ljust(width)
                continue
            cell = f"{case['min_ms']:.0f}"
            if i > 0 and base["min_ms"]:
                cell += f" (×{case['min_ms'] / base['min_ms']:.2f})"
            row += cell.ljust(width)
        print(row)
    print()
    return 0


# --------------------------------------------------------------------------- #
# Точка входа
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Бенчмарк скорости нарезки аудио через FFmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python benchmark/bench.py\n"
            "  python benchmark/bench.py --label home-pc\n"
            "  python benchmark/bench.py --quick\n"
            "  python benchmark/bench.py --compare\n"
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help="имя машины для файлов результатов (по умолчанию — имя хоста)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="повторов на случай (по умолчанию 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="прогревочных прогонов (по умолчанию 1)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="потоков для параллельного случая (по умолчанию — число ядер)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="быстрый прогон: 2 повтора, без прогрева",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="перегенерировать исходники в benchmark/.work/sources",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="не удалять срезы из benchmark/.work/slices после прогона",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="не запускать замеры, а сравнить готовые JSON в benchmark/results/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        return compare(list(RESULTS_DIR.glob("*.json")))

    repeats = 2 if args.quick else max(1, args.repeats)
    warmup = 0 if args.quick else max(0, args.warmup)
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
    label = (args.label or platform.node() or "unknown").strip()
    # Метка попадает в имя файла, поэтому убираем всё, что путает файловые системы.
    safe_label = re.sub(r"[^0-9A-Za-z._-]", "-", label) or "unknown"

    print("=" * 72)
    print("Бенчмарк нарезки аудио")
    print("=" * 72)

    try:
        ffmpeg = require_ffmpeg()
    except BenchError as exc:
        print(f"  [ERROR] {exc}")
        return 2

    env = collect_environment(ffmpeg)
    print(f"Машина      : {safe_label}")
    print(f"Процессор   : {env['cpu']} ({env['cpu_count']} логических ядер)")
    print(f"FFmpeg      : {ffmpeg}")
    print(f"Повторов    : {repeats} (прогрев {warmup}), потоков: {workers}")
    print()

    print("[1/3] Исходники")
    try:
        prepare_sources(args.regen)
    except BenchError as exc:
        print(f"  [ERROR] {exc}")
        return 1
    print()

    print("[2/3] Замеры")
    cases = build_cases(workers)
    results: dict[str, CaseResult] = {}
    started_all = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        print(f"  [{i:2d}/{len(cases)}] {case.title} ... ", end="", flush=True)
        try:
            result = measure(case, repeats, warmup)
        except BenchError as exc:
            print("сбой")
            print(f"  [ERROR] {case.name}: {exc}")
            return 1
        results[case.name] = result
        print(
            f"{result.min_ms:.0f} мс (медиана {result.median_ms:.0f}, "
            f"+{result.spread_pct:.0f}%)"
        )
    wall = time.perf_counter() - started_all
    print()

    print("[3/3] Отчёты")
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "label": safe_label,
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "environment": env,
        "config": {
            "repeats": repeats,
            "warmup": warmup,
            "workers": workers,
            "slice_count": SLICE_COUNT,
            "slice_duration_seconds": SLICE_DURATION,
            "long_source_seconds": LONG_DURATION,
            "short_source_seconds": SHORT_DURATION,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "pcm_codec": PCM_CODEC,
            "envelope_frame_samples": ENVELOPE_FRAME_SAMPLES,
        },
        "cases": [results[case.name].as_dict() for case in cases],
        "insights": derive_insights(results, workers),
        "wall_seconds": round(wall, 1),
    }

    json_path = RESULTS_DIR / f"{safe_label}.json"
    md_path = RESULTS_DIR / f"{safe_label}.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(f"  {rel(json_path)}")
    print(f"  {rel(md_path)}")
    print()

    if not args.keep_work:
        shutil.rmtree(SLICE_DIR, ignore_errors=True)

    ins = payload["insights"]
    print("-" * 72)
    print(
        f"Пустой запуск ffmpeg: {ins['startup_ms']:.0f} мс — "
        f"{ins['startup_share_of_slice_pct']:.0f}% времени одного среза."
    )
    print(
        f"Партия из {SLICE_COUNT} срезов: одним вызовом "
        f"×{ins['batch_single_speedup']:.1f}, в {workers} потоках "
        f"×{ins['batch_parallel_speedup']:.1f}."
    )
    print(f"Весь прогон занял {wall:.1f} с.")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
