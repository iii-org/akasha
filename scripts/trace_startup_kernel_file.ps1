param([string]$Output = "artifacts/startup-profile/kernel-file-new")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$directory = [IO.Path]::GetFullPath((Join-Path $root $Output))
if (Test-Path -LiteralPath $directory) { throw "Use a new directory: $directory" }
New-Item -ItemType Directory -Path $directory | Out-Null
$session = 'AkashaStartupIO-' + [guid]::NewGuid().ToString('N')
$trace = Join-Path $directory 'kernel-file.etl'
$owned = $false
$variables = @('PYTHONPYCACHEPREFIX','AKASHA_STARTUP_CACHE_OPS','PYTHONDONTWRITEBYTECODE','LANGSMITH_TRACING','LANGCHAIN_TRACING_V2')
$previous = @{}
foreach ($name in $variables) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:PYTHONPYCACHEPREFIX = '\\?\' + (Join-Path $directory 'cache-pyc')
    $env:AKASHA_STARTUP_CACHE_OPS = '1'
    $env:PYTHONDONTWRITEBYTECODE = $null
    $env:LANGSMITH_TRACING = 'false'
    $env:LANGCHAIN_TRACING_V2 = 'false'
    & logman create trace $session -o $trace -p '{EDD08927-9CC4-4E65-B970-C2560FB5C289}' 0xfff0 5 -bs 64 -nb 16 64 -f bincirc -max 256 -ets
    if ($LASTEXITCODE -ne 0) { throw 'ETW start failed' }
    $owned = $true
    & "$root/.venv/Scripts/python.exe" -u "$PSScriptRoot/profile_startup.py" --case akasha --trace-imports --threshold 120 --output $directory
    if ($LASTEXITCODE -ne 0) { throw 'Startup failed' }
} finally {
    if ($owned) {
        & logman query $session -ets | Out-File (Join-Path $directory 'collector-status.txt')
        & logman stop $session -ets
        if ($LASTEXITCODE -ne 0) { Write-Error "ETW stop failed; session name: $session" -ErrorAction Continue }
    }
    foreach ($name in $variables) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
$data = Get-Content (Join-Path $directory 'akasha-1.json') -Raw | ConvertFrom-Json
$targetPid = $data.import_trace.process_id
$xml = Join-Path $directory 'kernel-file-events.xml'
$writer = [IO.StreamWriter]::new($xml,$false,[Text.UTF8Encoding]::new($false))
try {
    $writer.WriteLine('<Events>')
    Get-WinEvent -Path $trace -Oldest -FilterXPath "*[System[Execution[@ProcessID='$targetPid']]]" | ForEach-Object { $writer.WriteLine($_.ToXml()) }
    $writer.WriteLine('</Events>')
} finally { $writer.Dispose() }
& "$root/.venv/Scripts/python.exe" "$PSScriptRoot/summarize_startup_io.py" $directory
& "$root/.venv/Scripts/python.exe" "$PSScriptRoot/correlate_startup_etw.py" (Join-Path $directory 'akasha-1.json') $xml
