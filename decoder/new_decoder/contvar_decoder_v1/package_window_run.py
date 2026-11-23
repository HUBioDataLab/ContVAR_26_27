"""Create a portable code/results backup after the full experiment is verified."""
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    verified = json.loads((root / "results/window_mean_seed42/verification.json").read_text(encoding="utf-8"))
    if not verified.get("all_checks_passed"):
        raise ValueError("Verify the full experiment before packaging")
    destination = root / "deliverables"
    destination.mkdir(exist_ok=True)
    archive = destination / "ContVAR_Mean_Window_Code_Results.zip"
    excluded = {".venv", "cache", "deliverables", "__pycache__", ".pytest_cache", ".git"}
    count = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(root.parent / "ContVAR.code-workspace", "ContVAR.code-workspace")
        z.writestr("README_ARCHIVE.txt", "ContVAR mean-window code, prepared cohorts, checkpoints, predictions and reports.\nOpen ContVAR.code-workspace and read contvar_decoder_v1/BASLA.md.\nLarge original encoder data folders are external and are not duplicated here.\nRestore ContVAR_Mean_Window_Cache.zip beside this archive to recover the exact mean-cache artifact.\n")
        for directory, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in excluded and name != "window_smoke_check"]
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix == ".pyc":
                    continue
                relative = path.relative_to(root)
                z.write(path, "contvar_decoder_v1/" + relative.as_posix(),
                        compress_type=zipfile.ZIP_STORED if path.suffix in (".zip", ".png") else zipfile.ZIP_DEFLATED)
                count += 1
        patch = destination / "CODE_CHANGES.patch"
        if patch.exists():
            z.write(patch, "contvar_decoder_v1/deliverables/CODE_CHANGES.patch")
    # Reading every archive member also validates the ZIP CRCs.
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"Archive CRC failure: {bad}")
    files = [archive, destination / "ContVAR_Mean_Window_Cache.zip"]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (destination / "SHA256SUMS_mean_window.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
    request = {"uploads": [{"local_path": str(path), "purpose": "create_library_file", "library_artifact_type": "other"} for path in files]}
    (destination / "library_upload_request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"code_results_files": count, "archives": [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashes[p.name]} for p in files]}))


if __name__ == "__main__":
    main()
