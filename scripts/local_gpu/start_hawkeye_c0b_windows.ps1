param(
    [int]$Port = 8000,
    [string]$Root = "",
    [int]$ContextSize = 8192,
    [string]$HostAddress = "100.83.147.56"
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    # Resolve from the data-drive registry: KINELO_DATA_ROOT\data_registry.json (default D:\).
    $dataRoot = if ($env:KINELO_DATA_ROOT) { $env:KINELO_DATA_ROOT } else { "D:\" }
    $registryPath = Join-Path $dataRoot "data_registry.json"
    $entry = (Get-Content -Raw -Encoding UTF8 $registryPath | ConvertFrom-Json).datasets.hawkeye_vlm
    if (-not $entry -or $entry.kind -ne "local") {
        throw "hawkeye_vlm is not registered as a local path in $registryPath"
    }
    $Root = Join-Path $dataRoot $entry.path
}

$server = Join-Path $Root "llama-b10423-vulkan\llama-server.exe"
$model = Join-Path $Root "models\Qwen3-VL-4B-Instruct-GGUF\Qwen3VL-4B-Instruct-Q4_K_M.gguf"
$mmproj = Join-Path $Root "models\Qwen3-VL-4B-Instruct-GGUF\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
$adapter = Join-Path $Root "adapters\hawkeye-c0b-seed42-f16.gguf"
$logDir = Join-Path $Root "logs"

foreach ($path in @($server, $model, $mmproj, $adapter)) {
    if (-not (Test-Path $path)) {
        throw "Required file is missing: $path"
    }
}

$existing = Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
    Where-Object { $_.CommandLine -like "*hawkeye-c0b-seed42*" }
if ($existing) {
    Write-Output "HawkEye C0B server is already running (PID $($existing.ProcessId))."
    exit 0
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "hawkeye-c0b.stdout.log"
$stderr = Join-Path $logDir "hawkeye-c0b.stderr.log"

$arguments = @(
    "--model", $model,
    "--mmproj", $mmproj,
    "--lora", $adapter,
    "--alias", "hawkeye-c0b-seed42",
    "--host", $HostAddress,
    "--port", "$Port",
    "--ctx-size", "$ContextSize",
    "--parallel", "1",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--n-gpu-layers", "99",
    "--jinja",
    "--temp", "0",
    "--top-k", "1",
    "--top-p", "1"
)

$startParams = @{
    FilePath = $server
    ArgumentList = $arguments
    RedirectStandardOutput = $stdout
    RedirectStandardError = $stderr
    PassThru = $true
}
$process = Start-Process @startParams

Write-Output "Started HawkEye C0B server (PID $($process.Id), port $Port)."
Write-Output "Logs: $stdout and $stderr"
