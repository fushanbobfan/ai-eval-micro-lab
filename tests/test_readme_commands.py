import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_SETTING = "src"
EVALUATOR = "python -m ai_eval_micro_lab.cli examples/quality-gate.jsonl"
FIRST_BLOCK = [
    f'$env:PYTHONPATH = "{PATH_SETTING}"',
    EVALUATOR,
    "python -m unittest discover -s tests",
]
BASH_SETTING = f"In bash, the first line is `export PYTHONPATH={PATH_SETTING}`."


class ReadmeFirstBlockTests(unittest.TestCase):
    """Compare the README's first fenced code block, and the bash sentence right after it,
    with the copy above, and run the copy's evaluator command from the repository root.

    The evaluator runs in a new process, since calling main() directly skips the module's
    __main__ guard. The unittest line stays out: run here, it would run this test again.
    """

    def test_readme_shows_these_commands(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        # Either fence CommonMark allows, indented up to three spaces, opens the first block.
        first_fence = re.search(r"^ {0,3}(```|~~~)", readme, re.MULTILINE)
        lines = readme[first_fence.start():].splitlines()
        message = "README's first block changed; try it in a clean clone and update this test"
        self.assertEqual(lines[:6], ["```powershell", *FIRST_BLOCK, "```", ""], message)
        self.assertEqual(lines[6][: len(BASH_SETTING)], BASH_SETTING, message)

    def test_evaluator_command_prints_the_report(self):
        result = subprocess.run(
            [sys.executable, *EVALUATOR.split()[1:]],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": PATH_SETTING},
            capture_output=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertTrue(result.stdout.strip(), "the command printed nothing")
        report = json.loads(result.stdout)
        # examples/quality-gate.jsonl has one exact match and token F1 scores of 1, 0.8 and 0.
        self.assertEqual(report["count"], 3)
        self.assertAlmostEqual(report["exact_match"], 1 / 3)
        self.assertAlmostEqual(report["token_f1"], 0.6)


if __name__ == "__main__":
    unittest.main()
