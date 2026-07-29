"""Модель данных сценария.

Модуль отвечает только за разбор ``scenario/timeline.json`` и приведение его
к типизированному виду. Он не знает ни про FFmpeg, ни про содержимое ассетов,
поэтому его можно тестировать и читать независимо от рендера.

Требуется Python 3.11+ (используется синтаксис ``X | None``).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ниже этого уровня считаем сигнал тишиной: в dB-огибающей нельзя
# использовать -inf, поэтому берём практический пол.
SILENCE_DB = -96.0


def configure_console() -> None:
    """Переводит вывод в UTF-8.

    По умолчанию консоль Windows использует cp1252/cp866, и любое русское
    сообщение падает с UnicodeEncodeError. Вызывается из точек входа
    (``build.py``, ``validator.py``) до первого print.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # перенаправленный или закрытый поток
            pass

_TIMECODE_RE = re.compile(r"^(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?$")


class ScenarioError(Exception):
    """Сценарий синтаксически или структурно некорректен."""


def parse_timecode(value: object, where: str) -> float:
    """Преобразует ``MM:SS.mmm`` (или число секунд) в секунды типа float.

    Примеры: ``"01:29.000"`` -> 89.0, ``"00:39.300"`` -> 39.3, ``12.5`` -> 12.5.
    """
    if isinstance(value, bool):
        raise ScenarioError(f"{where}: таймкод не может быть булевым значением")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ScenarioError(f"{where}: таймкод должен быть строкой 'MM:SS.mmm' или числом")

    match = _TIMECODE_RE.match(value.strip())
    if not match:
        raise ScenarioError(
            f"{where}: не удалось разобрать таймкод {value!r} "
            f"(ожидается формат 'MM:SS.mmm', например '01:29.000')"
        )
    minutes, seconds, millis = match.groups()
    if int(seconds) > 59:
        raise ScenarioError(f"{where}: в таймкоде {value!r} секунды больше 59")
    total = int(minutes) * 60 + int(seconds)
    if millis:
        total += int(millis.ljust(3, "0")) / 1000.0
    return float(total)


def format_timecode(seconds: float) -> str:
    """Обратное преобразование секунд в ``MM:SS.mmm`` — для отчётов и логов."""
    if seconds < 0:
        return f"-{format_timecode(-seconds)}"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:06.3f}"


@dataclass(frozen=True)
class VolumePoint:
    """Точка автоматизации громкости: время (сек) и уровень (dB)."""

    t: float
    db: float


@dataclass(frozen=True)
class PanPoint:
    """Точка автоматизации панорамы: время (сек) и положение от -1.0 до 1.0."""

    t: float
    pan: float


@dataclass
class DuckSpec:
    """Событие само приглушает указанные группы вокруг себя.

    Нужно для акцентов: выстрел или ледяной удар должны читаться отчётливо,
    поэтому фон на короткое время уходит вниз. В отличие от приглушения под
    голосами, окно задаётся явно (``pre`` до начала и ``hold`` после), а не
    длиной файла — у выстрела длинный реверб-хвост, и держать фон приглушённым
    всё это время не нужно.
    """

    groups: list[str]
    depth_db: float
    pre: float = 0.0
    hold: float = 0.3


@dataclass
class OutputConfig:
    """Параметры итогового файла и нормализации громкости."""

    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 24
    target_lufs: float = -16.0
    target_true_peak: float = -1.5
    target_lra: float = 11.0
    # loudnorm держит True Peak «около» заданного значения и обычно
    # промахивается вверх на несколько сотых dB. Запас гарантирует, что
    # итоговый замер окажется НЕ ВЫШЕ target_true_peak.
    true_peak_margin_db: float = 0.3
    # Предохранитель: если черновик почти тихий (например, ещё нет голосов),
    # буквальная нормализация до target_lufs подняла бы фон зала на +20 dB и
    # больше. Приросты выше этого значения обрезаются с предупреждением.
    max_normalize_gain_db: float = 14.0
    # Суффикс имён результатов по умолчанию для этого сценария: "v2" даёт
    # master_v2.wav. Смысл в том, чтобы каждый сценарий объявлял свои имена сам
    # и запуск без аргументов не затирал результаты другого сценария.
    # Аргумент --output-suffix перебивает это значение.
    default_suffix: str = ""
    # true — не давать loudnorm уходить в динамический режим.
    #
    # loudnorm применяет один постоянный коэффициент (linear) только пока
    # прирост укладывается в потолок True Peak. Иначе он молча переключается
    # на динамическую нормализацию и переформовывает динамику: тихие места
    # поднимаются сильнее громких. С этим флагом прирост ограничивается так,
    # чтобы режим остался линейным, а уровни сценария сохранились точно —
    # ценой того, что итоговая громкость окажется ниже target_lufs.
    preserve_dynamics: bool = False

    @property
    def pcm_codec(self) -> str:
        codecs = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
        if self.bit_depth not in codecs:
            raise ScenarioError(
                f"output.bit_depth={self.bit_depth} не поддерживается "
                f"(доступны: {sorted(codecs)})"
            )
        return codecs[self.bit_depth]

    @property
    def channel_layout(self) -> str:
        return {1: "mono", 2: "stereo"}.get(self.channels, f"{self.channels}c")


@dataclass
class DuckingConfig:
    """Параметры приглушения фонов под голосами."""

    attack: float = 0.08
    release: float = 0.30
    # Имя группы -> глубина приглушения в dB.
    groups: dict[str, float] = field(default_factory=lambda: {"ambience": 6.0, "crowd": 8.0})

    def depth_for(self, group: str | None) -> float:
        if not group:
            return 0.0
        return float(self.groups.get(group, 0.0))


@dataclass
class MixConfig:
    """Запас по уровню для событий с ``normalize_source``.

    Абсолютная шкала ставится не в 0 dBFS, а на ``absolute_headroom_db`` ниже.
    Смысл: при нормализации master фильтр loudnorm добавляет общий прирост, и
    если удар стоит слишком близко к 0 dBFS, лимитер True Peak расплющивает все
    удары в один уровень — заданный баланс до готового файла не доживает.

    Сдвиг общий для всех таких событий, поэтому относительный баланс между ними
    не меняется: значения dB в сценарии остаются такими, как в задании.
    """

    absolute_headroom_db: float = 0.0
    note: str = ""


@dataclass
class LoopConfig:
    """Параметры бесшовного зацикливания длинных фонов."""

    crossfade: float = 0.4
    curve: str = "qsin"  # qsin = equal-power кроссфейд


@dataclass
class TransitionConfig:
    """Описание перехода «зал -> замороженный зал» в районе 01:29."""

    fade_end: float = 88.85
    gap_ms: int = 150
    ice_event_id: str = "frozen_hall"
    note: str = ""


@dataclass
class Event:
    """Одно событие таймлайна: файл, его место во времени и его громкость."""

    id: str
    stem: str
    file: Path
    start: float
    end: float | None = None
    required: bool = False
    loop: bool = False
    duck_group: str | None = None
    gain_db: float = 0.0
    pan: float = 0.0
    # true — исходник сначала приводится к пику 0 dBFS, поэтому gain_db и
    # volume_points читаются как АБСОЛЮТНЫЙ целевой уровень, а не как
    # относительное ослабление. Нужно там, где собственный уровень файла
    # сильно отличается от остальных.
    normalize_source: bool = False
    volume_points: list[VolumePoint] = field(default_factory=list)
    # Автоматизация панорамы. Если задана, перебивает статический pan.
    pan_points: list[PanPoint] = field(default_factory=list)
    # Короткие фейды против щелчков на стыках (5-20 мс), в секундах.
    fade_in: float = 0.0
    fade_out: float = 0.0
    # true — обрезать тишину по краям исходника. Порог в начале -50 dB,
    # в конце -60 dB: хвост реверберации громче и не срезается.
    trim_silence: bool = False
    # Расширение стереобазы: 1.0 — как есть, больше — шире.
    width: float = 1.0
    # Поменять каналы местами. Нужно, когда в исходнике уже записано движение
    # панорамы, а по сценарию оно должно идти в другую сторону: встроенный образ
    # бывает сильнее любой автоматизации баланса, и перебить его нельзя.
    swap_channels: bool = False
    # Приглушение, которое это событие накладывает на другие группы.
    ducks: DuckSpec | None = None
    slot_hint: tuple[float, float] | None = None
    note: str = ""

    # Заполняется рендерером после probe исходника.
    source_duration: float | None = None
    resolved_end: float | None = None
    # Сколько секунд отрезано по краям при trim_silence.
    trim_start: float = 0.0
    trim_end: float = 0.0

    @property
    def has_fixed_end(self) -> bool:
        """True, если длительность задана сценарием, а не длиной файла."""
        return self.end is not None

    def window(self) -> tuple[float, float]:
        """Фактический интервал события. Доступно после resolve в рендерере."""
        if self.resolved_end is None:
            raise ScenarioError(f"событие {self.id!r}: длительность ещё не вычислена")
        return self.start, self.resolved_end


@dataclass
class Timeline:
    """Полный разобранный сценарий."""

    total_duration: float
    output: OutputConfig
    ducking: DuckingConfig
    loop: LoopConfig
    mix: MixConfig
    transition: TransitionConfig
    events: list[Event]
    stems: list[str]
    source_path: Path

    def events_for_stem(self, stem: str) -> list[Event]:
        return [e for e in self.events if e.stem == stem]


def _parse_volume_points(raw: object, where: str) -> list[VolumePoint]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioError(f"{where}.volume_points должен быть массивом")
    points: list[VolumePoint] = []
    for i, item in enumerate(raw):
        loc = f"{where}.volume_points[{i}]"
        if not isinstance(item, dict):
            raise ScenarioError(f"{loc} должен быть объектом с полями 't' и 'db'")
        if "t" not in item or "db" not in item:
            raise ScenarioError(f"{loc}: обязательны поля 't' и 'db'")
        db = item["db"]
        if not isinstance(db, (int, float)) or isinstance(db, bool):
            raise ScenarioError(f"{loc}.db должен быть числом")
        points.append(VolumePoint(t=parse_timecode(item["t"], f"{loc}.t"), db=float(db)))
    points.sort(key=lambda p: p.t)
    return points


def _parse_pan_points(raw: object, where: str) -> list[PanPoint]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioError(f"{where}.pan_points должен быть массивом")
    points: list[PanPoint] = []
    for i, item in enumerate(raw):
        loc = f"{where}.pan_points[{i}]"
        if not isinstance(item, dict) or "t" not in item or "pan" not in item:
            raise ScenarioError(f"{loc}: обязательны поля 't' и 'pan'")
        pan = item["pan"]
        if not isinstance(pan, (int, float)) or isinstance(pan, bool):
            raise ScenarioError(f"{loc}.pan должен быть числом")
        if not -1.0 <= float(pan) <= 1.0:
            raise ScenarioError(f"{loc}.pan={pan} вне диапазона [-1.0, 1.0]")
        points.append(PanPoint(t=parse_timecode(item["t"], f"{loc}.t"), pan=float(pan)))
    points.sort(key=lambda p: p.t)
    return points


def _parse_ducks(raw: object, where: str) -> DuckSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}.ducks должен быть объектом")
    groups = raw.get("groups")
    if not isinstance(groups, list) or not groups or not all(
        isinstance(g, str) for g in groups
    ):
        raise ScenarioError(f"{where}.ducks.groups должен быть непустым массивом строк")
    depth = raw.get("depth_db")
    if not isinstance(depth, (int, float)) or isinstance(depth, bool):
        raise ScenarioError(f"{where}.ducks.depth_db должен быть числом")
    if float(depth) < 0:
        raise ScenarioError(
            f"{where}.ducks.depth_db={depth} должен быть положительным — "
            f"это глубина приглушения, знак добавляется сам"
        )
    return DuckSpec(
        groups=[str(g) for g in groups],
        depth_db=float(depth),
        pre=float(raw.get("pre", 0.0)),
        hold=float(raw.get("hold", 0.3)),
    )


def _parse_event(raw: object, index: int, project_root: Path) -> Event:
    where = f"events[{index}]"
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where} должен быть объектом")

    for required_key in ("id", "stem", "file", "start"):
        if required_key not in raw:
            raise ScenarioError(f"{where}: отсутствует обязательное поле {required_key!r}")

    event_id = raw["id"]
    if not isinstance(event_id, str) or not event_id.strip():
        raise ScenarioError(f"{where}.id должен быть непустой строкой")

    file_value = raw["file"]
    if not isinstance(file_value, str) or not file_value.strip():
        raise ScenarioError(f"{where}.file должен быть непустой строкой")
    file_path = Path(file_value.replace("\\", "/"))
    if file_path.is_absolute():
        raise ScenarioError(
            f"{where}.file={file_value!r}: путь должен быть относительным "
            f"относительно корня проекта ({project_root})"
        )

    slot_hint = None
    if raw.get("slot_hint") is not None:
        hint = raw["slot_hint"]
        if not isinstance(hint, list) or len(hint) != 2:
            raise ScenarioError(f"{where}.slot_hint должен быть массивом из двух таймкодов")
        slot_hint = (
            parse_timecode(hint[0], f"{where}.slot_hint[0]"),
            parse_timecode(hint[1], f"{where}.slot_hint[1]"),
        )

    duck_group = raw.get("duck_group")
    if duck_group is not None and not isinstance(duck_group, str):
        raise ScenarioError(f"{where}.duck_group должен быть строкой или null")

    pan = raw.get("pan", 0.0)
    if not isinstance(pan, (int, float)) or isinstance(pan, bool):
        raise ScenarioError(f"{where}.pan должен быть числом")
    if not -1.0 <= float(pan) <= 1.0:
        raise ScenarioError(f"{where}.pan={pan} вне допустимого диапазона [-1.0, 1.0]")

    gain_db = raw.get("gain_db", 0.0)
    if not isinstance(gain_db, (int, float)) or isinstance(gain_db, bool):
        raise ScenarioError(f"{where}.gain_db должен быть числом")

    return Event(
        id=event_id,
        stem=str(raw["stem"]),
        file=file_path,
        start=parse_timecode(raw["start"], f"{where}.start"),
        end=parse_timecode(raw["end"], f"{where}.end") if raw.get("end") is not None else None,
        required=bool(raw.get("required", False)),
        loop=bool(raw.get("loop", False)),
        duck_group=duck_group,
        gain_db=float(gain_db),
        pan=float(pan),
        normalize_source=bool(raw.get("normalize_source", False)),
        volume_points=_parse_volume_points(raw.get("volume_points"), where),
        pan_points=_parse_pan_points(raw.get("pan_points"), where),
        fade_in=float(raw.get("fade_in", 0.0)),
        fade_out=float(raw.get("fade_out", 0.0)),
        trim_silence=bool(raw.get("trim_silence", False)),
        width=float(raw.get("width", 1.0)),
        swap_channels=bool(raw.get("swap_channels", False)),
        ducks=_parse_ducks(raw.get("ducks"), where),
        slot_hint=slot_hint,
        note=str(raw.get("note", "")),
    )


def load_timeline(path: Path, project_root: Path) -> Timeline:
    """Читает и валидирует структуру сценария (но не сами аудиофайлы)."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"не удалось прочитать сценарий {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{path}: некорректный JSON — {exc}") from exc

    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: корневой элемент должен быть объектом")
    if "events" not in data or not isinstance(data["events"], list):
        raise ScenarioError(f"{path}: обязательное поле 'events' должно быть массивом")

    out_raw = data.get("output", {})
    if not isinstance(out_raw, dict):
        raise ScenarioError(f"{path}: 'output' должен быть объектом")
    output = OutputConfig(
        sample_rate=int(out_raw.get("sample_rate", 48000)),
        channels=int(out_raw.get("channels", 2)),
        bit_depth=int(out_raw.get("bit_depth", 24)),
        target_lufs=float(out_raw.get("target_lufs", -16.0)),
        target_true_peak=float(out_raw.get("target_true_peak", -1.5)),
        target_lra=float(out_raw.get("target_lra", 11.0)),
        true_peak_margin_db=float(out_raw.get("true_peak_margin_db", 0.3)),
        max_normalize_gain_db=float(out_raw.get("max_normalize_gain_db", 14.0)),
        default_suffix=str(out_raw.get("default_suffix", "")).strip(),
        preserve_dynamics=bool(out_raw.get("preserve_dynamics", False)),
    )
    output.pcm_codec  # ранняя проверка bit_depth

    duck_raw = data.get("ducking", {})
    if not isinstance(duck_raw, dict):
        raise ScenarioError(f"{path}: 'ducking' должен быть объектом")
    groups_raw = duck_raw.get("groups", {"ambience": 6.0, "crowd": 8.0})
    if not isinstance(groups_raw, dict):
        raise ScenarioError(f"{path}: 'ducking.groups' должен быть объектом")
    ducking = DuckingConfig(
        attack=float(duck_raw.get("attack", 0.08)),
        release=float(duck_raw.get("release", 0.30)),
        groups={str(k): float(v) for k, v in groups_raw.items()},
    )

    loop_raw = data.get("loop", {})
    if not isinstance(loop_raw, dict):
        raise ScenarioError(f"{path}: 'loop' должен быть объектом")
    loop = LoopConfig(
        crossfade=float(loop_raw.get("crossfade", 0.4)),
        curve=str(loop_raw.get("curve", "qsin")),
    )

    mix_raw = data.get("mix", {})
    if not isinstance(mix_raw, dict):
        raise ScenarioError(f"{path}: 'mix' должен быть объектом")
    mix = MixConfig(
        absolute_headroom_db=float(mix_raw.get("absolute_headroom_db", 0.0)),
        note=str(mix_raw.get("note", "")),
    )
    if mix.absolute_headroom_db > 0:
        raise ScenarioError(
            f"{path}: mix.absolute_headroom_db={mix.absolute_headroom_db} должен быть "
            f"нулём или отрицательным — это запас вниз от 0 dBFS"
        )

    trans_raw = data.get("transition", {})
    if not isinstance(trans_raw, dict):
        raise ScenarioError(f"{path}: 'transition' должен быть объектом")
    transition = TransitionConfig(
        fade_end=parse_timecode(trans_raw.get("fade_end", "01:28.850"), "transition.fade_end"),
        gap_ms=int(trans_raw.get("gap_ms", 150)),
        ice_event_id=str(trans_raw.get("ice_event_id", "frozen_hall")),
        note=str(trans_raw.get("note", "")),
    )

    events = [_parse_event(raw, i, project_root) for i, raw in enumerate(data["events"])]

    stems_raw = data.get("stems", ["ambience", "voices", "sfx", "music"])
    if not isinstance(stems_raw, list) or not all(isinstance(s, str) for s in stems_raw):
        raise ScenarioError(f"{path}: 'stems' должен быть массивом строк")

    return Timeline(
        total_duration=parse_timecode(data.get("total_duration", 103.0), "total_duration"),
        output=output,
        ducking=ducking,
        loop=loop,
        mix=mix,
        transition=transition,
        events=events,
        stems=list(stems_raw),
        source_path=path,
    )
