"""Compile local modules and render architecture previews without training."""
import ast
from pathlib import Path

from .model import PairedGODecoder
from .report import WINDOWS, diagram
from .util import WORKSPACE, run_conditions, write_json


def main():
    checked = []
    for path in list((WORKSPACE / "contvar_mf").glob("*.py")) + list((WORKSPACE / "tests").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked.append(str(path.relative_to(WORKSPACE)))
    specs = []
    for c in run_conditions()[:4] + run_conditions()[8:16]:
        model = PairedGODecoder(head=c["head"], pair_mode=c["pair_mode"], window_mode=c["window_mode"])
        specs.append(dict(model_spec=model.spec, parameters=sum(p.numel() for p in model.parameters()),
                          local_comparison_tokens=(1 if c["window_mode"] == "pair_mean" else
                                                   7 if c["window_mode"] == "esim" else 0)))
    write_json(WORKSPACE / "verification/model_specs.json", specs)
    out = WORKSPACE / "verification/architecture_previews"
    out.mkdir(exist_ok=True, parents=True)
    import matplotlib.pyplot as plt
    for mode in WINDOWS:
        fig = diagram(mode)
        fig.savefig(out / f"{mode}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    write_json(WORKSPACE / "verification/code_review.json", dict(compiled=checked, training_runs=0))
    print("COMPILE_AND_DIAGRAM_PREVIEWS_OK")


if __name__ == "__main__":
    main()
