#!/usr/bin/env python3
"""
Export a W&B report or panel URL to PDF using headless Chrome/Chromium.

Examples:
    python updated_version/tools/export_wandb_report_pdf.py \
      --url "https://wandb.ai/<entity>/<project>/reports/<slug>" \
      --output wandb_report.pdf

    # If the report is private, reuse a local Chrome profile.
    # Close Chrome first if you point at your active profile directory.
    python updated_version/tools/export_wandb_report_pdf.py \
      --url "https://wandb.ai/<entity>/<project>/reports/<slug>" \
      --output wandb_report.pdf \
      --user-data-dir "$HOME/Library/Application Support/Google/Chrome" \
      --profile-directory "Default"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


COMMON_BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_browser(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        raise FileNotFoundError(f"Browser not found: {explicit}")

    for candidate in COMMON_BROWSERS:
        if Path(candidate).exists():
            return candidate

    for candidate in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "Could not find Chrome/Chromium. Pass --browser with an explicit path."
    )


def build_command(args: argparse.Namespace, browser: str) -> list[str]:
    output_path = str(Path(args.output).expanduser().resolve())

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={args.window_width},{args.window_height}",
        f"--virtual-time-budget={args.virtual_time_budget_ms}",
        f"--print-to-pdf={output_path}",
    ]

    if args.no_sandbox:
        cmd.append("--no-sandbox")

    if args.user_data_dir:
        cmd.append(f"--user-data-dir={Path(args.user_data_dir).expanduser()}")

    if args.profile_directory:
        cmd.append(f"--profile-directory={args.profile_directory}")

    cmd.append(args.url)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a W&B report/panel URL to PDF with headless Chrome."
    )
    parser.add_argument("--url", required=True, help="W&B report or panel URL")
    parser.add_argument(
        "--output",
        required=True,
        help="Output PDF path, for example reports/wandb_report.pdf",
    )
    parser.add_argument(
        "--browser",
        default=None,
        help="Optional explicit Chrome/Chromium binary path",
    )
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help="Chrome user data dir to reuse an authenticated session",
    )
    parser.add_argument(
        "--profile-directory",
        default=None,
        help='Chrome profile name, for example "Default" or "Profile 1"',
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Browser width in pixels",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2200,
        help="Browser height in pixels",
    )
    parser.add_argument(
        "--virtual-time-budget-ms",
        type=int,
        default=15000,
        help="Wait budget before printing, in milliseconds",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Pass --no-sandbox to Chrome if needed",
    )
    args = parser.parse_args()

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        browser = find_browser(args.browser)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    cmd = build_command(args, browser)

    print("[INFO] Browser:", browser)
    print("[INFO] Output :", out_path)
    print("[INFO] URL    :", args.url)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "\n[ERROR] PDF export failed.\n"
            "If your W&B report is private, try --user-data-dir and --profile-directory.\n"
            "If you use your active Chrome profile, close Chrome first.\n",
            file=sys.stderr,
        )
        return exc.returncode or 1

    if out_path.exists():
        print(f"[DONE] PDF saved to {out_path}")
        return 0

    print("[ERROR] Chrome exited but the PDF was not created.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
