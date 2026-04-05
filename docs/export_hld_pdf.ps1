param(
    [string]$HtmlPath = "docs/DiabetesInsight_HLD.html",
    [string]$OutputPath = "docs/DiabetesInsight_HLD.pdf"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$fullHtmlPath = (Resolve-Path $HtmlPath).Path
$outputDir = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$fullOutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputPath))

$browserCandidates = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe"
)

$browserPath = $browserCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browserPath) {
    throw "No compatible browser found for headless PDF export."
}

$htmlUri = [System.Uri]::new($fullHtmlPath).AbsoluteUri
$tempProfile = Join-Path $env:TEMP ("diabetes-insight-pdf-" + [Guid]::NewGuid().ToString("N"))

try {
    $args = @(
        "--headless",
        "--disable-gpu",
        "--disable-crash-reporter",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        """--user-data-dir=$tempProfile""",
        """--print-to-pdf=$fullOutputPath""",
        """$htmlUri"""
    )

    $process = Start-Process -FilePath $browserPath -ArgumentList ($args -join " ") -NoNewWindow -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Browser export failed with exit code $($process.ExitCode)."
    }

    if (-not (Test-Path $fullOutputPath)) {
        throw "PDF export did not create the expected file: $fullOutputPath"
    }

    Write-Output "PDF created at $fullOutputPath"
}
finally {
    if (Test-Path $tempProfile) {
        Remove-Item -LiteralPath $tempProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
}
