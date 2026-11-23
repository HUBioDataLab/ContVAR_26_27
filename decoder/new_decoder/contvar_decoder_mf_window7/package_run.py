"""Package this new run only; raw input embeddings and the large cache stay external."""
from pathlib import Path
import zipfile
import json

ROOT=Path(__file__).resolve().parent
qa=json.loads((ROOT / "verification/final_qa.json").read_text(encoding="utf-8"))
if not qa["passed"] or qa["checked_models"]!=48:
    raise RuntimeError("Complete verified run required")
destination=ROOT / "deliverables"
destination.mkdir(exist_ok=True)
archive=destination / "ContVAR_MF_Window7_Code_Results.zip"
included=[]
for folder in ["contvar_mf","tests","reports","verification","results","prepared","logs"]:
    for path in (ROOT / folder).rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in [".h5",".partial"] or path.name=="last.pt":
            continue
        included.append(path)
included += [ROOT / n for n in ["README.md","DURUM.md","AGENTS.md","run_all.ps1","offline_run.py","package_run.py","requirements.txt"]]
included += [ROOT / "ontology" / n for n in ["go-basic.json","go-basic.source.json"]]
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in included:
        z.write(path,path.relative_to(ROOT))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(included)),ensure_ascii=False))
