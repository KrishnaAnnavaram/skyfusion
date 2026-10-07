# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
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
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of skyfusion. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **station** | The OpenWeather History Bulk observations of one site (`st_*` columns) | ground truth, sensor |
| **POWER** | The NASA POWER hourly reanalysis and satellite data (`pw_*` columns) | satellite (alone), NASA data |
| **source** | The station or POWER | dataset, feed |
| **UTC grid** | The regular hourly index in UTC that all columns share | timeline, calendar |
| **LST** | Local standard time of a POWER file: UTC + round(lon / 15) hours, no daylight saving | local time (alone) |
| **fill value** | The POWER code `-999` for a missing value | sentinel, null |
| **fusion rule** | A pair of columns with the same physical quantity that skyfusion averages (`fz_*`) | merge, blend |
| **target** | The observed station temperature `target_temp_c` | label, fused temperature |
| **mask column** | A `*_missing` column that is 1 where the source value was missing | flag, indicator |
| **origin** | The last observed hour of a sample | anchor, start time |
| **window** | The W hours of inputs that end at the origin | lookback, sequence |
| **horizon** | The number of hours h after the origin | lead time, step |
| **direct forecast** | One model output for all horizons 1 to H | multi-output, recursive forecast |
| **split** | The parts train, validation and test, by year of the origin | fold |
| **embargo** | The removal of samples whose target hours cross into the next part | gap, purge |
| **baseline** | Persistence, seasonal naive or climatology | naive model, dummy |
| **skill** | 1 − RMSE / RMSE of persistence, at one horizon | improvement, gain |
| **feature set** | `station` (station columns) or `fused` (station, POWER and `fz_*` columns) | input set, mode |
| **ablation** | The comparison of the two feature sets on identical samples | experiment, A/B test |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **align** | Put both sources on the UTC grid |
| **shift** | Change LST times to UTC |
| **fuse** | Average two columns under a fusion rule |
| **fill** | Copy the last value forward, for max 3 hours, in a feature column |
| **check** | Set a value outside its physical limits to NaN |
| **split** | Assign samples to train, validation and test by year |
| **fit** | Learn parameters and preprocessing from the train samples |
| **select** | Choose a hyperparameter with the validation samples |
| **score** | Calculate MAE, RMSE and skill on the test samples |
