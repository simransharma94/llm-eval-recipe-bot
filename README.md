# LLM Eval Practice: Recipe Bot

A small learning project: how to evaluate an LLM app, from reading outputs by hand to automated checks.

The "product" is a recipe assistant with these rules:

- Only answer cooking questions, and politely decline everything else
- Keep every answer under 150 words
- Ask one clarifying question when a request is vague or contradictory
- Don't invent details such as serving counts
- Reply in Hinglish when the user writes in Hinglish
- Refuse to drop these rules when the user says "ignore your rules"

## Why evals

LLM outputs vary, a prompt fix can quietly break something else, and nobody can read every output once real users arrive. An eval is a repeatable way to check whether the product does what you intended.

The project follows the "three gulfs" framework from Hamel Husain and Shreya Shankar:

| Gulf | The gap | What I did about it |
|---|---|---|
| Comprehension | You can't read all your data, so you don't know how it fails | Read every output by hand and wrote a note on each failure |
| Specification | Your prompt doesn't say what you actually mean | Turned the failure notes into explicit rules and examples |
| Generalization | It works on your examples but not on everything users send | Held-out questions, 3 runs per question, automated checks |

## What I did

1. **Manual error analysis.** 21 test questions (normal, constrained, vague, off-topic, long, Hinglish, contradictory), a pass/fail and a note for each answer, in a spreadsheet.
2. **Grouped failures into modes and counted them.** Too long was the biggest, then guessing instead of asking, answering off-topic questions, obeying "ignore your rules", and wrong formats.
3. **Fixed the prompt.** With no instructions, 3 of 21 answers passed. After adding rules and examples, 14 of 21.
4. **Automated the checks** in `eval.py` and validated them against my own labels (they agreed on 19 of 21).
5. **Added held-out questions and ran every question 3 times.** This exposed over-refusal (declining valid cooking questions) and flaky answers that single runs hid.

## Results

| Stage | Result |
|---|---|
| No instructions (claude.ai) | 3/21 |
| Rules added (claude.ai) | 14/21 |
| Automated runner, prompt v3 (Gemini) | 27/31 |
| Automated runner, prompt v4, 3 runs per question (Gemini) | 108/111 |

These runs used different models and a growing test set, so they are not a like-for-like benchmark. Treat them as a record of the learning process.

## Files

| File | Purpose |
|---|---|
| `eval.py` | Runner with automated checks. Standard library only, no installs. |
| `test_cases.csv` | Test questions with the expected behavior (answer / clarify / decline / refuse a long request), split into `dev` and `heldout` |
| `prompt_v3.txt`, `prompt_v4.txt` | System prompts, one per iteration |
| `v2_outputs.csv` | Answers I labeled by hand, used to validate the checks |

## Automated checks

- **length**: at most 150 words
- **behavior**: the reply is the expected kind (answer, clarify, decline, or refuse a long request)
- **servings**: no invented serving count when the user gave none
- **language**: Hinglish questions get a Hinglish reply (rough word-list check)

## How to run

Requires Python 3. No packages to install.

```bash
# Score answers collected by hand (no API key needed)
python eval.py --score-only v2_outputs.csv

# Dry run of the whole loop with a fake bot that declines everything (no key)
python eval.py --provider mock

# Run the real bot, 3 times per question, with a free Gemini key
export GEMINI_API_KEY="your-key"
python eval.py --prompt prompt_v4.txt --provider gemini --model gemini-3.5-flash-lite --runs 3 --delay 5

# Re-run only some cases
python eval.py --prompt prompt_v4.txt --provider gemini --model gemini-3.5-flash-lite --runs 5 --ids 8,9,H12
```

Model names change often. If you get a 404 error, check the provider's current model list and pass `--model`. Never commit an API key.

## Limitations

- The 21 `dev` questions were used to tune the prompts. The `heldout` questions are a better signal, but there are only a few.
- The checks are pattern matching and rely on the exact decline wording in the prompt. They can't judge quality. One run passed an answer that said "bake for 20 minutes per person".
- There is no LLM-as-judge yet for fuzzy criteria like "no unrequested extras".
- Scope choices (for example, food chemistry is out of scope) were my own, and the prompt and test labels have to match.

## Next steps

- Add an LLM judge for subjective criteria, validated against human labels
- Collect new, messier inputs (typos, odd phrasing) and add them to the test set
- Run on a larger held-out set and report flaky cases
