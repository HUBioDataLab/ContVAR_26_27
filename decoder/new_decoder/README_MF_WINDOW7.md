# ContVAR MF / window-7 source export

This directory adds the completed MF/window-7 decoder experiments to
`decoder/new_decoder/`. The current implementation is
`contvar_decoder_mf_window7/contvar_mf/`.

The `contvar_decoder_v1/` source is retained for the historical implementation
and the eight exact-logit parity checks in the current test suite. It is the
handover-based working implementation, including the mean-window extension;
this export does not claim that it is a separate teammate's exact code version.

## Included records

- Current and historical decoder source, tests, scripts and requirements.
- The 48-run result JSON files, configurations, training histories and source snapshots.
- Aggregate CSV tables, figures, PDF/HTML/Markdown reports and existing verification records.
- Preparation manifest and the exact GO ontology source URL/hash metadata.

Original source and experiment records are copied byte-for-byte. Scientific
model and training code has not been modified for this export. Existing reports
and verification records describe the complete original local experiment, not
a new training run performed during export.

## External artifacts

Raw input datasets/embeddings, prepared HDF5 features, row-level TSV files,
model weights, Python environments, logs and delivery ZIPs are not included.
Reading the reports and running the current 24 unit/protocol tests does not
require those large data artifacts. Training, checkpoint inference, complete
independent evaluation and regenerating the full code/results ZIP do require
the corresponding original artifacts.

The expected sibling data directory is
`New_Decoder_Data_with_local_Context/`, next to the two decoder directories.
Restore the original `ontology/go-basic.json` matching the SHA-256 recorded in
`contvar_decoder_mf_window7/ontology/go-basic.source.json` for data preparation.
The current ontology download can differ from the frozen original release.

## Windows environment and test command

From this directory, create the environment expected by the existing runner:

```powershell
py -3.14 -m venv contvar_decoder_v1/.venv
& ./contvar_decoder_v1/.venv/Scripts/python.exe -m pip install -r ./contvar_decoder_mf_window7/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
Set-Location ./contvar_decoder_mf_window7
& ../contvar_decoder_v1/.venv/Scripts/python.exe -I -S ./offline_run.py pytest tests -q -p no:cacheprovider
```

The requirements record the original Windows/CUDA environment. A different
platform needs the appropriate PyTorch build. The PowerShell runner and
`offline_run.py` use the sibling `contvar_decoder_v1/.venv` deliberately;
no virtual environment is committed.

Full experiments use `run_all.ps1` after restoring the data artifacts. This
export does not start training. The original readmes retain their historical
completion notes and paths; this document states what this Git export contains.
