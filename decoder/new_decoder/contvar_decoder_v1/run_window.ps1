param(
    [ValidateSet('prepare','test','smoke','train','report')]
    [string]$Action = 'test',
    [string]$Out = '',
    [ValidateSet('cuda','cpu')]
    [string]$Device = 'cuda',
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Once BASLA.md dosyasindaki kurulumu tamamlayin.' }
    switch ($Action) {
        'test' { & $pythonExe -m pytest tests -q }
        'prepare' { & $pythonExe -m contvar_decoder.window_prepare }
        'smoke' {
            if (-not $Out) { $Out = 'results/window_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
            & $pythonExe -m contvar_decoder.window_suite --smoke --cohorts old --families hard --device $Device --seed $Seed --out $Out
        }
        'train' {
            if (-not $Out) { $Out = 'results/window_mean_seed' + $Seed + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
            & $pythonExe -m contvar_decoder.window_suite --device $Device --seed $Seed --out $Out
        }
        'report' {
            if (-not $Out) { throw 'Raporlanacak deney klasorunu -Out ile belirtin.' }
            & $pythonExe -m contvar_decoder.window_report $Out
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
