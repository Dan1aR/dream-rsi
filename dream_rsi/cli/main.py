# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DREAM-RSI unified CLI."""

from __future__ import annotations

import argparse
import sys

from dream_rsi.cli import explore, evolve, learning
from dream_rsi.cli._common import install_sigint_handler, load_dotenv


SECTIONS = ("explore", "learning", "evolve")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dream-rsi",
        description="DREAM-RSI: recursive self-improvement through evolving worlds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  dream-rsi explore init -w workspace/lasso\n"
            "  dream-rsi explore run -w workspace/lasso -a gemini --branches 10 --refine 10\n"
            "  dream-rsi explore archive -w workspace/lasso --cycle 0\n"
            "  dream-rsi learning discover -w workspace/lasso -a gemini --rounds 3\n"
            "  dream-rsi learning directions -w workspace/lasso -a gemini --cycle 0\n"
            "  dream-rsi evolve -w workspace/lasso -a gemini --cycles 5\n"
        ),
    )
    sub = p.add_subparsers(dest="section", metavar="SECTION", required=True)

    sub.add_parser(
        "explore",
        help="online exploration: init | run | archive (see: dream-rsi explore run --help)",
    )
    sub.add_parser(
        "learning",
        help="offline learning: replay | export | allocate | discover | directions",
    )
    sub.add_parser("evolve", help="full loop: explore → learning → promote, repeated")

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    section = argv[0]
    rest = argv[1:]

    if section not in SECTIONS:
        build_parser().print_help()
        print(f"\nunknown section: {section}", file=sys.stderr)
        return 2

    if section == "explore":
        install_sigint_handler()
        return explore.main(rest)
    if section == "learning":
        return learning.main(rest)
    return evolve.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
