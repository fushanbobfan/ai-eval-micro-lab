# AI Eval Micro Lab

Small, dependency-free evaluation tools for comparing model predictions with reference answers.

The first experiment provides normalized exact match and token-level F1 for JSONL datasets. Each line should contain `expected` and `predicted` strings.

```powershell
python -m ai_eval_micro_lab.cli examples.jsonl
python -m unittest discover -s tests
```

This repository is intended for reproducible learning experiments. Future additions should include tests, a short explanation of the idea, and a runnable example.

