# Writing Style for This Project

Style instructions for `research-report.md` and any other prose in this repository (READMEs, data-processing-pipeline.md, etc.). Follow these for all writing tasks unless explicitly overridden.

## Reference Voice

Write in the style of Päivi Majaranta's HCI papers (e.g. her CHI 2009 gaze-typing work, COGAIN 2009 text-editing paper). Plain, direct, no hedging.

## Structure

- Standard HCI paper structure: Abstract → Introduction → Method (Participants, Apparatus, Procedure, Task) → Analysis → Results → Discussion → Acknowledgments → References.
- Roman numerals for main sections (I, II, III...).
- Title Case headers as short noun phrases.
- No colons with subtitle descriptions in headers. Short: *Random-window Baseline*, not *Random-window Baseline: A Within-Participant Permutation Check for RQ1*.
- Results subsection headers mirror Analysis subsection headers.

## Voice and Tone

- Plain declarative sentences. Short paragraphs, usually 2–5 sentences. One idea per sentence; split if a sentence has two.
- Passive voice and "we"-voice both acceptable; mix naturally. First-person singular ("I") is allowed sparingly, only for author-preference or author-choice statements (e.g. *"I prefer this stricter approach because..."*). Default is passive or "we".
- Flat, direct limitation statements. No hedging. Just state it.
- Acknowledge what the work doesn't do without apologizing.
- Write for a fresh reader. No meta-framing ("unlike the previous version").
- Jargon only where unavoidable; define on first use.
- Avoid nominalizations and "per X" qualifiers when a direct rephrase works. Cut any phrase whose removal does not change the meaning.

**This is the default for every sentence, not just rewrites.** Don't wait to be told. Before adding text, read the surrounding paragraphs and match their cadence.

## Paragraph Flow

A paragraph has a single arc. Before writing, pick a flow and follow it end to end. General rule: **claim first, justification second**. Don't open with a justification and end with the claim. If three unrelated observations land in a row, split the paragraph or reorder it.

## Numbers and Stats

- Inline with the sentence: *"59% of turns contained a shift"*, not *"the percentage was 59%"*.
- Report means with SD in parentheses. For proportions presented as percentages, use percentage form throughout: *"59.4% (14.5%)"* (not *"0.59 (0.15)"*). Do not mix decimal and percent form for the same quantity.
- Percentage-point differences use "pp": *"+22.4 pp"*, not *"+22.4%"*.

## Formatting

- **No em-dashes or en-dashes anywhere.** Use commas, parentheses, or colons instead.
- Minimal bold/italic. Italicize only defined terms on first use and figure/table labels.
- No bullet lists in running prose; use tables for tabular data and prose for everything else.
- Figure and table captions start with **Figure N.** or **Table N.** in bold, followed by a short descriptive sentence.
- Code references in markdown use `[filename:LINE](relative/path#LLINE)` format.

## Spelling

**US English throughout.** Notable conversions:
- behavior (not behaviour)
- analyze (not analyse)
- normalize (not normalise)
- anthropomorphize (not anthropomorphise)
- stabilize (not stabilise)
- labeling (not labelling)
- operationalization (not operationalisation)
- summarize (not summarise)

## References to Prior Work

- Be precise about what the cited study actually measured. Do not round or generalize a finding into a broader claim the paper did not make.
- When a cited result is from a different task or operationalization, note the difference clearly.
- Example: Kendrick and Holler measured *polar (yes-no)* responses specifically — don't paraphrase as "any response".

## Project Terminology (Canonical)

These are the defined terms in this project. Use them consistently across the report, figures, and scripts.

**Aggregated AOI (binary, per sample):**
- **on-robot** — sample hit the Face AOI
- **away** — sample hit Body or Outside but not Face (formerly *off-robot*; do not use that form)
- The six drawn AOIs (Face, Eyes, Mouth, Nose, Body, Outside) are collapsed into one binary per-sample variable called *the aggregated AOI*. A sample's aggregated AOI is either *on-robot* or *away*. Do not append *label*, *state*, or *group* after *aggregated AOI*: the phrase *the aggregated AOI* already names the per-sample variable.
- AOI vs AOIs: *AOI* (singular) when naming one specific region (e.g. *the Face AOI*) or when referring to the aggregated AOI as a single variable. *AOIs* (plural) when counting or listing several (e.g. *the six drawn AOIs*, *the Eyes, Mouth, and Nose sub-AOIs*).
- Reserved senses: do not use *group* for on-robot/away (the word *group* in this report means a run of consecutive same-label samples in RQ1's event-detection rule, or a cohort-level aggregate). Do not use *state* for on-robot/away either: *state* is reserved for Kendon's methodological sense (*state measurement* vs *event measurement*).
- *Aggregated AOI* overlaps with a Tobii Pro Lab feature name (Pro Lab manual §9.9.2), but this study does not use that Pro Lab feature; the collapse is done in Python in step1_preprocess.py.

**Per-turn event metrics (RQ1):**
- **look-away** at turn start — do not use *claim* or *turn-claiming* as the name for this metric; use *look-away*.
- **look-back** at turn end — do not use *yield* or *turn-yielding* as the name for this metric; use *look-back*.
- The phrases *turn-claiming signal* and *turn-yielding signal* are fine as theoretical labels for what a gaze shift is proposed to do in conversation (e.g. *"the shift is the turn-claiming signal"*). Just don't rename the metric.

**Metric outputs:**
- A participant's RQ1(a)/(b) **result** (not *score*).

**Turn events and windows:**
- **turn start** = first spoken word of a turn
- **turn end** = last spoken word of a turn
- **window** = the ±1 s region around a turn start or turn end used in RQ1
- Prefer *window* over *turn boundary* when referring to the analysis region. *Turn boundary* is avoided because it overlaps conceptually with *window*. If referring specifically to the start or end point, say *turn start* or *turn end*, not *turn boundary*.
- Use *turn start* / *turn end* when describing this study's data. *Turn onset* is fine when describing the broader HHI literature (e.g. *"turn-onset gaze-aversion rates reported for human-to-human conversation"*).

**Research questions** (verbatim from §I of the report):
- **RQ1, Turn-taking regulation.** (a) How often do participants look away from the robot at turn start? (b) How often do they look back at the robot at turn end?
- **RQ2, Listening vs Speaking.** (a) What percentage of time is spent looking at the robot in each role? (b) Do participants look at the robot more while listening than while speaking?

**Technical:**
- **LLM system-prompt** (hyphenated), not *LLM prompt*.

## Report Structure Conventions

**Data Pre-processing** (§III.A) uses bold inline titles per paragraph:
- *Mapping gaze from recordings to the snapshot*
- *Areas of Interest*
- *Aggregated AOI* (introduces the two aggregated AOIs, *on-robot* and *away*)
- *Gaze data export*
- *Transcript and turn events* (not *turn boundaries*: avoid the word *boundaries* entirely)

**Apparatus** (§II) uses bold inline subsection titles:
- *Location*
- *Robot*
- *Eye tracker*

**RQ definitions** use this header style:
```
**RQ1(a). Defining 'Look-away at turn start'.** A *look-away at turn start* is …
```
Bold the whole header, italicize the defined term on first use.

**Procedure** (§II) splits activities into short paragraphs, one per main activity (glasses prep, calibration, move to test room, consent and task, post-task). No multi-clause run-on sentences.

**Subsection titles.**
- *Results* (§IV) subsection titles prefix the RQ: *RQ1. Turn-taking Gaze Behavior*, *RQ2. On-robot Proportion by Listening vs Speaking*.
- *Analysis* (§III) subsection titles use *RQ1 Operationalization*, *RQ2 Operationalization*. "Operationalization" is used instead of "Analysis" to avoid repeating the parent section title.
- No colon-subtitle descriptions in any header.

## Appendix Conventions

Short and pointer-style. The Appendix is a cross-reference, not a narrative.

- Pointer form in main text: direct and specific. Cite the appendix subsection by letter and say what is there. *"See Appendix B for the processed gaze data."* Do not use the older *"[X] is included in the Appendix. Refer to it for [Y]."* template: it is indirect and does not tell the reader where to go.
- Scripts: point to the GitHub repo URL and the `data-processing-pipeline.md` walkthrough; do not list each script in running prose.
- Data: describe as one unified per-sample gaze dataset (AOI hits + pupil + turn annotations), plus one per-participant results CSV. Do not split into four separate file callouts.

## When Touching Figures or Scripts

- Keep figure labels consistent with the report terminology. When terminology changes, regenerate figures from scripts.
- Script comments and docstrings use the same canonical terms (on-robot/away, look-away/look-back, RQ1/RQ2).
- Figure captions start with **Figure N.** bold label.
