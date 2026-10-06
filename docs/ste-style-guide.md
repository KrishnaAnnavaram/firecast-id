# ASD-STE100 Simplified Technical English: the standard for this repository

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. Rules for the text

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the firecast-id documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **detection** | One FIRMS row: one active-fire pixel at one time. | hotspot (for one row), fire point, pixel |
| **hotspot count** | The number of kept detections on one day. The forecast target. | fire count, hotspot number |
| **kept detection** | A detection with a kept `type` and a confidence at or above the limit. | valid fire, clean row |
| **fire type** | The FIRMS `type` value: 0, 1, 2 or 3. | class, category |
| **daily series** | The file with one row per calendar day: `date`, `count`, `frp_sum`. | time series (for the file), dataset |
| **zero day** | A day with no kept detection. Its count is 0. | missing day, empty day |
| **climate driver** | A monthly value such as ONI, DMI or rainfall. | covariate (in prose), exogenous variable |
| **publication lag** | The days after month end before a climate value is public. | delay, latency |
| **origin** | The last day with known counts when a forecast is made. | anchor, cutoff (in prose) |
| **horizon** | The number of days from the origin to the target date. | lead time, step |
| **target date** | Origin + horizon. | forecast day, future day |
| **window** | The lag values that end at the origin, with the target of one horizon. | sequence (in prose), sample |
| **feature row** | One row of features for one origin and one horizon. | instance, record |
| **model** | One forecast method from the model list. | algorithm, network (for all models) |
| **baseline** | The model `persistence`, `seasonal_naive` or `climatology`. | benchmark model, naive model |
| **search space** | The parameter values that tuning can select for a model. | grid (in prose), hyperparameter set |
| **trial** | One parameter set that tuning fits and scores. | run, experiment |
| **test year** | The calendar year that the backtest forecasts and scores. | holdout, evaluation year |
| **validation year** | The year before the test year. Only tuning uses it. | dev year, tuning set |
| **backtest** | The year-by-year test of all models, horizons and test years. | cross-validation, evaluation (for the procedure) |
| **reference** | The baseline that the DM test compares each model with. | benchmark, control |
| **DM test** | The Diebold-Mariano test on the daily absolute errors of two models. | significance test (alone), t-test |
| **peak season** | The months in `FIRECAST_PEAK_MONTHS`. | fire season, dry season (for the setting) |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **clean** | Remove detections with a bad value, a removed fire type or a low confidence. |
| **count** | Calculate the number of kept detections per calendar day. |
| **fetch** | Download detections from the FIRMS API. |
| **build** | Make windows and feature rows from the daily series. |
| **tune** | Fit and score the trials on the validation year and keep the best one. |
| **fit** | Train a model on feature rows. |
| **forecast** | Give the expected hotspot count for a target date. |
| **score** | Calculate the metrics from forecasts and the counts of their target dates. |
