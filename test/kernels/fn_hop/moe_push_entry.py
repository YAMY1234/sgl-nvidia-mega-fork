"""Optional isolated checker around the MoE-hop driver, on every rank."""
import os
from pathlib import Path
import sys

command = [sys.executable, str(Path(__file__).with_name("moe_push_driver.py")), *sys.argv[1:]]
tool = os.environ.get("FN_HOP_SANITIZER")
if tool:
    options = ["--tool", tool, "--report-api-errors", "no", "--error-exitcode", "99"]
    if tool == "racecheck":
        options += ["--racecheck-report", "hazard", "--show-backtrace", "yes"]
    command = ["compute-sanitizer", *options, *command]
os.execvp(command[0], command)
