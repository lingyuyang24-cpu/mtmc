param(
    [bool]$Display = $false,
    [int]$MaxFrames = 0,
    [string]$RunTag = 'stability_v3'
)

$ErrorActionPreference = 'Stop'
$python = 'E:\anaconda\envs\pytorch\python.exe'
$project = $PSScriptRoot
$videoRoot = 'D:\mtmc\videos\init'
$outputRoot = Join-Path $project 'videos\output'
$streams = @(
    (Join-Path $videoRoot 'view-HC2.mp4')
    (Join-Path $videoRoot 'view-HC3.mp4')
)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python 不存在：$python"
}

$invalidFileNameChars = '[\\/:*?"<>|\s]'
$inputTag = (($streams | ForEach-Object {
    [System.IO.Path]::GetFileNameWithoutExtension($_)
}) -join '_') -replace $invalidFileNameChars, '_'
$safeRunTag = $RunTag -replace $invalidFileNameChars, '_'
$startedAt = Get-Date
$timestamp = $startedAt.ToString('yyyyMMdd_HHmmss_fff')
$runName = "${inputTag}_${safeRunTag}_${timestamp}"
$runDir = Join-Path $outputRoot $runName

$trackingOutput = Join-Path $runDir 'tracking.avi'
$globalLogOutput = Join-Path $runDir 'global_decisions.jsonl'
$trackLogOutput = Join-Path $runDir 'tracks.jsonl'
$runInfoOutput = Join-Path $runDir 'run_info.json'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$arguments = @(
    (Join-Path $project 'demo_stream.py')
    '--streams'
) + $streams + @(
    '--stream-mode', 'queue'
    '--stream-queue-size', '2'
    '--detector', (Join-Path $project 'yolo11l.pt')
    '--detector-score', '0.20'
    '--detector-imgsz', '1280'
    '--reid-backend', 'transreid'
    '--transreid-download', 'false'
    '--encoder-batch-size', '2'
    '--deep-sort-max-cosine-distance', '0.20'
    '--tracker-max-age', '30'
    '--track-duplicate-containment', '0.85'
    '--track-duplicate-max-cosine-distance', '0.15'
    '--post-nms', 'nms'
    '--nms-max-overlap', '0.35'
    '--tracker-new-track-min-confidence', '0.25'
    '--global-feature-min-confidence', '0.60'
    '--global-feature-min-box-height', '96'
    '--global-feature-max-occlusion', '0.20'
    '--global-feature-update-max-distance', '0.35'
    '--global-reid-threshold', '0.35'
    '--global-reid-strong-threshold', '0.25'
    '--global-reid-margin', '0.02'
    '--global-gallery-match', 'adaptive'
    '--global-candidate-threshold', '0.45'
    '--global-borderline-confirm-frames', '15'
    '--global-borderline-confirm-ratio', '0.80'
    '--global-borderline-confirm-threshold', '0.35'
    '--global-prototype-count', '6'
    '--global-prototype-merge-threshold', '0.15'
    '--same-camera-reconnect-reid-threshold', '0.50'
    '--same-camera-reconnect-reid-margin', '0.05'
    '--same-camera-reconnect-timeout', '1800'
    '--same-camera-reconnect-distance', '1.00'
    '--same-camera-reconnect-confirm-threshold', '0.50'
    '--same-camera-reconnect-confirm-ratio', '0.80'
    '--same-camera-conflict-continuity', '5.0'
    '--show-local-id', 'false'
    '--tile-width', '640'
    '--tile-height', '360'
    '--display', $Display.ToString().ToLowerInvariant()
    '--output', $trackingOutput
    '--debug-global-log', $globalLogOutput
    '--track-log', $trackLogOutput
)
if ($MaxFrames -gt 0) {
    $arguments += @('--max-frames', $MaxFrames.ToString())
}

$runInfo = [ordered]@{
    run_name = $runName
    run_tag = $RunTag
    started_at = $startedAt.ToString('o')
    input_files = $streams
    display = $Display
    max_frames = $MaxFrames
    outputs = [ordered]@{
        video = $trackingOutput
        global_decisions = $globalLogOutput
        tracks = $trackLogOutput
    }
    command_arguments = $arguments
}
$runInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runInfoOutput -Encoding UTF8

Write-Host "本次运行：$runName"
Write-Host "输出目录：$runDir"

Push-Location $project
try {
    & $python @arguments
} finally {
    Pop-Location
}
