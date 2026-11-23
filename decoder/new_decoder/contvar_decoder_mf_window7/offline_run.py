"""Run a local module with Python network calls denied before third-party imports.

Usage: python -I -S offline_run.py --log logs/run.log MODULE [ARGS...]
The isolated interpreter skips user startup/site hooks. Only the audited local
venv's package directory is added, without executing .pth startup files.
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
for key, value in {
    "POWERSHELL_TELEMETRY_OPTOUT": "1", "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    "VSCODE_TELEMETRY_LEVEL": "off", "DO_NOT_TRACK": "1", "OTEL_SDK_DISABLED": "true",
    "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "MPLCONFIGDIR": str(ROOT / "runtime_cache/matplotlib"),
    "TORCH_HOME": str(ROOT / "runtime_cache/torch"),
}.items():
    os.environ[key] = value


def deny_network(event, arguments):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind",
                 "socket.sendto", "socket.sendmsg"}:
        raise PermissionError(f"Offline experiment runner denied network event: {event}")


sys.dont_write_bytecode = True
sys.addaudithook(deny_network)
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "contvar_decoder_v1/.venv/Lib/site-packages"))
arguments = sys.argv[1:]
if arguments[:1] == ["--log"]:
    log = ROOT / arguments[1]
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = log.open("a", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    arguments = arguments[2:]
if not arguments:
    raise SystemExit("Specify a local module")
module, args = arguments[0], arguments[1:]
if module == "--self-test":
    import socket
    try:
        socket.getaddrinfo("example.invalid", 443)
    except PermissionError:
        print("OFFLINE_GUARD_OK: DNS/network attempt denied before access")
    else:
        raise RuntimeError("Offline guard failed")
else:
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__")
