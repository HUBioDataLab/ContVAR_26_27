param(
    [ValidateSet("all","prepare","check","fit","evaluate","report","verify","package")]
    [string]$Stage = "all",
    [ValidateSet("cuda","cpu")]
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$taskRoot = $PSScriptRoot
$taskPython = Join-Path (Split-Path -Parent $taskRoot) "contvar_decoder_v1\.venv\Scripts\python.exe"
$env:POWERSHELL_TELEMETRY_OPTOUT = "1"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
Set-Location -LiteralPath $taskRoot
function Invoke-OfflineModule {
    param([string]$Module,[string]$Log,[string[]]$ModuleArgs=@())
    & $taskPython -I -S (Join-Path $taskRoot "offline_run.py") --log $Log $Module @ModuleArgs
    if ($LASTEXITCODE -ne 0) { throw "$Module failed. Inspect $Log. Exit: $LASTEXITCODE" }
}
if ($Stage -eq "all" -or $Stage -eq "prepare") {
    if (-not (Test-Path -LiteralPath (Join-Path $taskRoot "prepared\manifest.json"))) {
        Invoke-OfflineModule -Module "contvar_mf.prepare" -Log "logs/preparation.log"
    }
}
if ($Stage -eq "all" -or $Stage -eq "check") {
    Invoke-OfflineModule -Module "pytest" -Log "logs/tests.log" -ModuleArgs @("tests","-q","--junitxml=verification/tests.xml")
    Invoke-OfflineModule -Module "contvar_mf.validate_prepared" -Log "logs/preflight.log"
}
if ($Stage -eq "all") {
    Invoke-OfflineModule -Module "contvar_mf.suite" -Log "logs/suite.log" -ModuleArgs @("--phase","all","--device",$Device,"--resume")
} elseif ($Stage -eq "fit" -or $Stage -eq "evaluate") {
    Invoke-OfflineModule -Module "contvar_mf.suite" -Log "logs/suite.log" -ModuleArgs @("--phase",$Stage,"--device",$Device,"--resume")
}
if ($Stage -eq "all" -or $Stage -eq "report") {
    Invoke-OfflineModule -Module "contvar_mf.report" -Log "logs/report.log"
}
if ($Stage -eq "all" -or $Stage -eq "verify") {
    Invoke-OfflineModule -Module "contvar_mf.verify_run" -Log "logs/final_qa.log"
}
if ($Stage -eq "all" -or $Stage -eq "package") {
    Invoke-OfflineModule -Module "package_run" -Log "logs/package.log"
}
Write-Output "Completed stage: $Stage"
