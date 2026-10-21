param(
    [int]$Repeats = 2,
    [string]$Output = "artifacts/startup-profile/windows-io-20261009"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $root $Output))
if (Test-Path -LiteralPath $outputPath) { throw "Use a new output directory: $outputPath" }
New-Item -ItemType Directory -Path $outputPath | Out-Null
$previousPrefix = $env:PYTHONPYCACHEPREFIX
$previousOps = $env:AKASHA_STARTUP_CACHE_OPS
$previousNoWrite = $env:PYTHONDONTWRITEBYTECODE
$previousSmith = $env:LANGSMITH_TRACING
$previousChain = $env:LANGCHAIN_TRACING_V2
$owned = $false
try {
    for ($run = 1; $run -le $Repeats; $run++) {
        $status = (& wpr -status 2>&1 | Out-String)
        if ($status -notmatch 'WPR is not recording') { throw "WPR may already be recording; refusing to interfere: $status" }
        $prefix = Join-Path $outputPath "run-$run-pyc"
        $env:PYTHONPYCACHEPREFIX = '\\?\' + $prefix
        $env:AKASHA_STARTUP_CACHE_OPS = '1'
        $env:PYTHONDONTWRITEBYTECODE = $null
        $env:LANGSMITH_TRACING = 'false'
        $env:LANGCHAIN_TRACING_V2 = 'false'
        $trace = Join-Path $outputPath "run-$run.etl"
        # Memory mode avoids continuous trace writes competing with the test.
        & wpr -start GeneralProfile -start FileIO -start Minifilter
        if ($LASTEXITCODE -ne 0) { throw "WPR start failed: $LASTEXITCODE" }
        $owned = $true
        try {
            & "$root/.venv/Scripts/python.exe" -u "$PSScriptRoot/profile_startup.py" --case akasha --run $run --trace-imports --threshold 120 --output $outputPath 2>&1 | Tee-Object -FilePath (Join-Path $outputPath "run-$run.log")
            if ($LASTEXITCODE -ne 0) { throw "Startup failed: $LASTEXITCODE" }
        } finally {
            & wpr -status 2>&1 | Out-File (Join-Path $outputPath "run-$run-wpr-status.txt")
            & wpr -stop $trace "Akasha offline cold startup run $run" -skipPdbGen
            if ($LASTEXITCODE -ne 0) { throw "WPR stop failed: $LASTEXITCODE" }
            $owned = $false
        }
    }
} finally {
    if ($owned) { & wpr -cancel }
    $env:PYTHONPYCACHEPREFIX = $previousPrefix
    $env:AKASHA_STARTUP_CACHE_OPS = $previousOps
    $env:PYTHONDONTWRITEBYTECODE = $previousNoWrite
    $env:LANGSMITH_TRACING = $previousSmith
    $env:LANGCHAIN_TRACING_V2 = $previousChain
}
