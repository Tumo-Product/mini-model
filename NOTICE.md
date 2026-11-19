# Data licence and attribution (read before using the tune data)

## Where the tunes come from

All training tunes come from **The Session** (https://thesession.org), via the weekly data dump at
https://github.com/adactio/TheSession-data (`json/tunes.json`). If you use it, please consider
[donating to the upkeep of The Session](https://thesession.org/donate).

The files in `data/` were built from that dump on 2026-11-10. `jigai prepare-data` rebuilds them from a fresh copy.

## The licence has an unusual clause

`TheSession-data/LICENSE.md` (checked 2026-11-11) is the **ODC Open Database License (ODbL)** plus
extra terms at the top. The extra terms include:

> **Prohibition on LLM Use:** You may not use, adapt, modify, or process the material in any way with
> Large Language Models. This includes but is not limited to training Large Language Models, utilizing
> LLM tools, or incorporating the material into any LLM-related applications or systems [...]

with an exception only for accessibility for disabled people.

## Permission from The Session

TUMO asked Jeremy Keith, who runs The Session, for permission to use The Session's tune data in this
activity, and how he would like The Session to be credited. He replied by email on 2026-11-17:

> No problem using the data from The Session.

So this project may use The Session's data for the activity, including training the model on it. The email is
kept by the TUMO team; ask them for a copy.

He didn't say how The Session should be credited, so we keep the ODbL credit below. The permission was asked
for and given for this activity: ask again before using the data for something else.

## ODbL obligations (if you share data or results)

- If you share a **database derived from it** (for example the files in `data/`), keep it under the ODbL,
  include the licence, and keep these notices.
- If you **publicly use a "Produced Work"**, for example generated tunes shown to people, include a notice like:
  *"Contains information from The Session, which is made available here under the Open Database License (ODbL)."*
- Using it internally within an organisation is not "public" use under the ODbL (section 4.5c).

## Model weights

The checkpoints in `weights/` (shared separately, not in git) were trained on this data. `tsALL-s6000-v1` is
covered by the permission above for use in the activity. `d482-s2400` was trained on 482 jigs from an older
2,000-tune book that isn't in this repo; its licence wasn't recorded, so check it before releasing it.
