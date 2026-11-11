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

What this means for this project, as we read it (not legal advice; ask whoever handles licensing at TUMO):

- The model here is a **10.8M-parameter character-level GPT**. It is far smaller than what is usually called
  a "large" language model, but it is the same kind of model, and the clause is written broadly.
  **Whether training it on The Session's data is allowed is an open question.** Decide before training.
- The clause also covers **processing the data with LLM tools**. That includes AI coding assistants
  (Claude, Copilot, ChatGPT) reading the tune files, and LLM-based tutors or chat features built on the data.
  Earlier experiments in this project did run AI agents over the corpus, which this clause may not allow.

## ODbL obligations (if you share data or results)

- If you share a **database derived from it** (for example the files in `data/`), keep it under the ODbL,
  include the licence, and keep these notices.
- If you **publicly use a "Produced Work"**, for example generated tunes shown to people, include a notice like:
  *"Contains information from The Session, which is made available here under the Open Database License (ODbL)."*
- Using it internally within an organisation is not "public" use under the ODbL (section 4.5c).
  The LLM clause, however, is not limited to public use.

## Model weights

The checkpoints in `weights/` (shared separately, not in git) were trained on this data, so the same questions
apply to them.
