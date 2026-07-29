# Голоса ElevenLabs: кого использовать для генерации

Кастинг реплик номера. Записано, чтобы при следующей генерации на
[elevenlabs.io](https://elevenlabs.io) не подбирать голоса заново и не получить
персонажа, который звучит иначе, чем в уже готовых файлах.

Тексты реплик — в [scenario/CLAUDE_HANDOFF_LOHEN_AUDIO_PROJECT.md](../scenario/CLAUDE_HANDOFF_LOHEN_AUDIO_PROJECT.md).
Куда класть готовые файлы — [README, раздел 3](../README.md).

## Кастинг

| Персонаж | Голос ElevenLabs | Тег подачи |
|---|---|---|
| Аукционист (ведущий) | **James — Husky, Engaging and Bold** | `[grand, theatrical, projecting to a large hall]` |
| Покупатель | **Serafina — Sensual Temptress** | `[quietly, suspicious]` |
| Лоэн | **Christian — Deep, Ominous, and Engaging** | `[slowly, quietly, with a sly undertone, almost under his breath]` |

Теги в квадратных скобках — указания подачи для модели v3. Они заметно влияют на
результат: у James реплика вышла на 8–9 dB громче остальных именно из-за
`projecting to a large hall`, а у Serafina — самой тихой из-за `quietly`. Если
генерировать остальные реплики без тегов, баланс между персонажами не совпадёт с
уже готовыми файлами.

## Что уже сгенерировано

Сессия от 2026-07-24 18:13:57, два дубля (Generation 1 и 2). Взят дубль 2 — он
медленнее и с более выраженными паузами.

| Событие | Голос | Текст как сгенерирован |
|---|---|---|
| `auctioneer_intro` | James | `Ladies and gentlemen... our final lot of the evening.` |
| `buyer_suspicious` | Serafina | `He looks far too calm.` |
| `lohen_notice` | Christian | `I was beginning to think... that no one... would notice.` |

Две мелочи, где сгенерированный текст отличается от сценария:

* `auctioneer_intro` — это сокращённая версия, без второй фразы
  «A weapon without equal.» Сценарий такой вариант прямо разрешает.
* `lohen_notice` — в сценарии «I was beginning to think... no one would
  notice.», сгенерировано «...**that** no one... would notice.» с двумя
  многоточиями вместо одного. На слух это добавляет вторую паузу.

## Что осталось сгенерировать

| Событие | Голос | Текст из сценария |
|---|---|---|
| `auctioneer_sold` | James | `Sold.` |
| `lohen_reveal` | Christian | `You misunderstand.` / `I was never the merchandise.` |
| `lohen_final_question` | Christian | `Was that all?` |
| `lohen_final_line` | Christian | `How disappointing.` |

Для `lohen_final_question` и `lohen_final_line` сценарий требует спокойной
подачи без крика, с холодным разочарованием — тег Лоэна подходит как есть.
Для `auctioneer_sold` уместнее укоротить тег до `[loud, final, decisive]`:
«Sold.» — удар, а не объявление в зал.

## Как переносить результат в проект

**1. Экспорт.** ElevenLabs отдаёт MP3 128 kbps, моно, 44.1 кГц. Это потери, а
голоса несут весь номер — если в тарифе есть экспорт в WAV/PCM, берите его.

**2. Один файл на всю сессию.** При нескольких спикерах выгружается **один**
файл, названный по одному из голосов — в нашем случае по Christian, хотя внутри
три разных голоса. По имени файла содержимое определить нельзя, надо смотреть
саму сессию.

**3. Разрезать на события.** Сценарий ждёт по одному файлу на событие с точными
именами. Блоки внутри выгрузки разделены **настоящей цифровой тишиной** около
−85 dB, тогда как паузы внутри реплики — это фон около −45 dB. Поэтому границы
надёжно ищутся так:

```bash
ffmpeg -hide_banner -i session.mp3 -af "silencedetect=noise=-60dB:d=0.25" -f null -
```

Паузы длиннее 0.4 с между блоками — границы реплик; более короткие принадлежат
самой реплике и разрезать по ним нельзя.

**4. Формат под проект.** Остальные ассеты — 48 кГц, стерео, `pcm_s16le`.
Приведение выглядит так (границы подставить свои):

```bash
ffmpeg -y -i session.mp3 -af "atrim=start=8.300:end=13.557,asetpts=PTS-STARTPTS,aresample=48000:first_pts=0,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,afade=t=in:st=0:d=0.008:curve=tri,afade=t=out:st=5.249:d=0.008:curve=tri" -ar 48000 -ac 2 -c:a pcm_s16le assets/voices/lohen_notice.wav
```

Фейды 8 мс по краям — против щелчков на стыках. Перевод моно в стерео забирает
около 3 dB: FFmpeg сохраняет суммарную мощность, а не уровень канала. Это не
потеря, но пики готовых файлов окажутся ниже, чем у исходного MP3.

## Уровни: что учесть

Голосовые события в сценарии, в отличие от всех 20 событий `sfx`, **не имеют**
ни `normalize_source`, ни `gain_db` — файл попадает в микс с коэффициентом 1.0.
Замерено на сборке с тремя готовыми репликами:

| | без голосов | с тремя голосами |
|---|---|---|
| Прирост `loudnorm` в линейном режиме | +8.7 dB | **+1.8 dB** |
| Crest микса | 16.4 dB | 17.6 dB |
| Громкость master | −18.6 LUFS | **−20.0 LUFS** |

То есть с добавлением голосов master стал **тише**, а не громче: пик голоса
съедает запас, из которого `loudnorm` берёт линейный прирост. Когда появятся
все семь реплик, голосовым событиям понадобится `normalize_source: true` плюс
`gain_db` — так же, как у эффектов, где общий запас задаёт
`mix.absolute_headroom_db`. Конкретные значения — вопрос баланса, его решать на
слух.
