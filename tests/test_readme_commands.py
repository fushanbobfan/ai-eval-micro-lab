"""Keep the README's commands working from the root of a clean clone.

Each command runs in a new process, since calling a module's main() directly skips its
__main__ guard. The README's unittest line runs in the workflow, before pytest is installed:
run from here, it would run this file again.
"""

import importlib
import json
import os
import pkgutil
import re
import subprocess
import sys
import unittest
from pathlib import Path

import ai_eval_micro_lab

ROOT = Path(__file__).resolve().parents[1]
PATH_SETTING = "src"
EVALUATOR = "python -m ai_eval_micro_lab.cli examples/quality-gate.jsonl"
TESTS = "python -m unittest discover -s tests"
FIRST_BLOCK = [
    f'$env:PYTHONPATH = "{PATH_SETTING}"',
    EVALUATOR,
    TESTS,
]
BASH_SETTING = f"In bash, the first line is `export PYTHONPATH={PATH_SETTING}`."


def run(command):
    """Run a README command line from the repository root with src on the import path."""
    return subprocess.run(
        [sys.executable, *command.split()[1:]],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": PATH_SETTING},
        capture_output=True,
        timeout=60,
    )


INTO_ARGUMENT = "a backtick right after an argument puts the line break into that argument"
ENDS_HERE = "a space after the backtick ends the command on this line"
PAST_THE_BLOCK = "the block ends after a backtick, so the command continues past the block"
# PowerShell passes these characters on to a native command unchanged, so splitting a command on
# whitespace gives the arguments Python receives. Others, such as ( ; $ @ { & and curly quotes,
# it reads as syntax.
PLAIN = r"\w \t./\\:=-"


def fenced_commands(readme):
    """Return [line number, fence label, command, problem] for each command in the README's
    fenced code blocks, read as plain text and joined the way PowerShell joins lines: a
    backtick after a space or tab at the end of a line continues the command on the next line,
    and a # at the start of a line or after whitespace starts a comment. `problem` names a line
    end PowerShell reads some other way.
    """
    commands, fence, continued = [], None, False
    for number, line in enumerate(readme.splitlines(), 1):
        if fence is None:
            opening = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
            if opening:
                fence, label, continued = opening.group(1), opening.group(2).strip(), False
            continue
        if re.fullmatch(rf" {{0,3}}{fence[0]}{{{len(fence)},}}[ \t]*", line):
            if continued:
                commands[-1][3] = commands[-1][3] or PAST_THE_BLOCK
            fence = None
            continue
        line = re.sub(r"(^|\s)#.*", r"\1", line)
        if continued:
            commands[-1][2] = commands[-1][2][:-1] + line
        elif line.strip():
            commands.append([number, label, line.strip(), None])
        else:
            continue
        continued = line.endswith("`")
        if re.search(r"\S`$", line):
            commands[-1][3] = commands[-1][3] or INTO_ARGUMENT
        elif re.search(r"`\s+$", line):
            commands[-1][3] = commands[-1][3] or ENDS_HERE
    return commands


class ReadmeFirstBlockTests(unittest.TestCase):
    """Compare the README's first fenced code block, and the bash sentence right after it,
    with the copy above, and check the report of the copy's evaluator command.
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
        result = run(EVALUATOR)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertTrue(result.stdout.strip(), "the command printed nothing")
        report = json.loads(result.stdout)
        # examples/quality-gate.jsonl has one exact match and token F1 scores of 1, 0.8 and 0.
        self.assertEqual(report["count"], 3)
        self.assertAlmostEqual(report["exact_match"], 1 / 3)
        self.assertAlmostEqual(report["token_f1"], 0.6)


class ReadmeCommandTests(unittest.TestCase):
    """Run every command in the README's fenced code blocks the way the first block sets them up.

    The first block's path line and unittest line are skipped. Any other command fails its
    subtest when it holds a character besides letters, digits, spaces and _ . / \\ : = -, and is
    otherwise split on whitespace, as PowerShell splits it.
    """

    def test_every_readme_command_runs(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        shown = set()
        for number, label, command, problem in fenced_commands(readme):
            module = re.match(r"python\s+-m\s+ai_eval_micro_lab\.(\w+)", command)
            if module:
                shown.add(module.group(1))
            with self.subTest(line=number):
                self.assertEqual(label, "powershell", "this test reads README blocks as PowerShell")
                if problem:
                    self.fail(f"{problem}: {command!r}")
                if " ".join(command.split()) in (FIRST_BLOCK[0], TESTS):
                    continue
                other = re.search(rf"[^{PLAIN}]", command)
                if other:
                    self.fail(
                        f"{other.group()!r} is outside the characters this test reads as"
                        " PowerShell does (letters, digits, spaces and _ . / \\ : = -):"
                        f" {command!r}"
                    )
                if not module:
                    self.fail(
                        "from a clean clone, commands run as python -m ai_eval_micro_lab.<module>;"
                        f" this line is {command!r}"
                    )
                result = run(command)
                stderr = result.stderr.decode(errors="replace")
                self.assertTrue(result.stdout.strip(), f"the command printed nothing\n{stderr}")
                report = json.loads(result.stdout)
                self.assertIsInstance(report, dict, "the command printed no JSON object")
                # Exit code 0 means every configured gate passed, and 1 that the report lists
                # the failures.
                expected = 1 if report.get("passed") is False else 0
                self.assertEqual(result.returncode, expected, stderr)

        commands = {
            info.name
            for info in pkgutil.iter_modules(ai_eval_micro_lab.__path__)
            if hasattr(importlib.import_module(f"ai_eval_micro_lab.{info.name}"), "main")
        }
        message = "every command needs a python -m example in a README block"
        self.assertEqual(shown, commands, message)


if __name__ == "__main__":
    unittest.main()
