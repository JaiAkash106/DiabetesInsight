param(
    [string]$MarkdownPath = "docs/detailed_documentation.md",
    [string]$OutputPath = "docs/DiabetesInsight_Detailed_Documentation.pdf"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Escape-PdfText {
    param([string]$Text)

    if ($null -eq $Text) {
        return ""
    }

    return $Text.Replace("\", "\\").Replace("(", "\(").Replace(")", "\)")
}

function Wrap-Text {
    param(
        [string]$Text,
        [int]$Width
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @("")
    }

    $words = $Text -split "\s+"
    $current = ""
    $lines = New-Object System.Collections.Generic.List[string]

    foreach ($word in $words) {
        if ([string]::IsNullOrWhiteSpace($word)) {
            continue
        }

        if (($current.Length + $word.Length + 1) -le $Width) {
            if ($current.Length -eq 0) {
                $current = $word
            } else {
                $current = "$current $word"
            }
            continue
        }

        if ($current.Length -gt 0) {
            $lines.Add($current)
        }

        $current = $word
    }

    if ($current.Length -gt 0) {
        $lines.Add($current)
    }

    return $lines.ToArray()
}

function Get-Style {
    param([string]$Line)

    if ($Line.StartsWith("### ")) {
        return @{
            Text = $Line.Substring(4)
            Font = "F2"
            Size = 13
            Leading = 18
            MaxChars = 70
        }
    }

    if ($Line.StartsWith("## ")) {
        return @{
            Text = $Line.Substring(3)
            Font = "F2"
            Size = 16
            Leading = 22
            MaxChars = 62
        }
    }

    if ($Line.StartsWith("# ")) {
        return @{
            Text = $Line.Substring(2)
            Font = "F2"
            Size = 18
            Leading = 24
            MaxChars = 56
        }
    }

    if ($Line.StartsWith("|")) {
        return @{
            Text = $Line
            Font = "F1"
            Size = 10
            Leading = 14
            MaxChars = 88
        }
    }

    if ($Line.StartsWith('```')) {
        return @{
            Text = "----------------------------------------"
            Font = "F1"
            Size = 10
            Leading = 14
            MaxChars = 88
        }
    }

    if ($Line.StartsWith("- ")) {
        return @{
            Text = $Line
            Font = "F1"
            Size = 11
            Leading = 16
            MaxChars = 80
        }
    }

    if ($Line -match '^\d+\. ') {
        return @{
            Text = $Line
            Font = "F1"
            Size = 11
            Leading = 16
            MaxChars = 80
        }
    }

    return @{
        Text = $Line
        Font = "F1"
        Size = 11
        Leading = 16
        MaxChars = 82
    }
}

function Add-ContentLine {
    param(
        [System.Collections.Generic.List[string]]$Page,
        [string]$Font,
        [int]$Size,
        [int]$X,
        [int]$Y,
        [string]$Text
    )

    $escaped = Escape-PdfText $Text
    $Page.Add("BT /$Font $Size Tf 1 0 0 1 $X $Y Tm ($escaped) Tj ET")
}

function Measure-CenterX {
    param(
        [string]$Text,
        [int]$Size
    )

    $pageWidth = 612
    $estimatedWidth = [math]::Min(480, [math]::Max(80, [math]::Round($Text.Length * $Size * 0.47)))
    return [math]::Max(60, [math]::Floor(($pageWidth - $estimatedWidth) / 2))
}

$fullMarkdownPath = Join-Path (Get-Location) $MarkdownPath
$fullOutputPath = Join-Path (Get-Location) $OutputPath

if (-not (Test-Path $fullMarkdownPath)) {
    throw "Markdown documentation file not found: $fullMarkdownPath"
}

$markdown = Get-Content $fullMarkdownPath -Raw -Encoding UTF8
$allLines = $markdown -split "`r?`n"

$bodyStart = 0
for ($i = 0; $i -lt $allLines.Count; $i++) {
    if ($allLines[$i] -eq "## 1. Executive Summary") {
        $bodyStart = $i
        break
    }
}

$bodyLines = $allLines[$bodyStart..($allLines.Count - 1)]
$pages = New-Object System.Collections.Generic.List[object]

$coverPage = New-Object System.Collections.Generic.List[string]
Add-ContentLine -Page $coverPage -Font "F2" -Size 26 -X (Measure-CenterX "DiabetesInsight" 26) -Y 690 -Text "DiabetesInsight"
Add-ContentLine -Page $coverPage -Font "F2" -Size 18 -X (Measure-CenterX "Detailed Project Documentation" 18) -Y 650 -Text "Detailed Project Documentation"
Add-ContentLine -Page $coverPage -Font "F1" -Size 15 -X (Measure-CenterX "Prepared by: Jai Akash" 15) -Y 570 -Text "Prepared by: Jai Akash"
Add-ContentLine -Page $coverPage -Font "F1" -Size 15 -X (Measure-CenterX "Program: B.Tech AIDS" 15) -Y 540 -Text "Program: B.Tech AIDS"
Add-ContentLine -Page $coverPage -Font "F1" -Size 15 -X (Measure-CenterX "Academic Level: 3rd Year" 15) -Y 510 -Text "Academic Level: 3rd Year"
Add-ContentLine -Page $coverPage -Font "F1" -Size 12 -X (Measure-CenterX "Generated on March 26, 2026" 12) -Y 430 -Text "Generated on March 26, 2026"
$pages.Add($coverPage) | Out-Null

$currentPage = New-Object System.Collections.Generic.List[string]
$y = 750

foreach ($rawLine in $bodyLines) {
    $line = $rawLine.TrimEnd()

    if ([string]::IsNullOrWhiteSpace($line)) {
        $y -= 10
        if ($y -lt 70) {
            $pages.Add($currentPage) | Out-Null
            $currentPage = New-Object System.Collections.Generic.List[string]
            $y = 750
        }
        continue
    }

    $style = Get-Style $line
    $segments = Wrap-Text -Text $style.Text -Width $style.MaxChars

    foreach ($segment in $segments) {
        if ($y -lt 70) {
            $pages.Add($currentPage) | Out-Null
            $currentPage = New-Object System.Collections.Generic.List[string]
            $y = 750
        }

        Add-ContentLine -Page $currentPage -Font $style.Font -Size $style.Size -X 54 -Y $y -Text $segment
        $y -= $style.Leading
    }
}

if ($currentPage.Count -gt 0) {
    $pages.Add($currentPage) | Out-Null
}

$objects = New-Object System.Collections.Generic.List[string]
$pageObjectNumbers = New-Object System.Collections.Generic.List[int]

$objects.Add("<< /Type /Catalog /Pages 2 0 R >>") | Out-Null
$objects.Add("") | Out-Null
$objects.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>") | Out-Null
$objects.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>") | Out-Null

$nextObjectNumber = 5
foreach ($page in $pages) {
    $stream = ($page -join "`n")
    $streamLength = [System.Text.Encoding]::ASCII.GetByteCount($stream)
    $pageObjectNumbers.Add($nextObjectNumber) | Out-Null
    $contentObjectNumber = $nextObjectNumber + 1

    $objects.Add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents $contentObjectNumber 0 R >>") | Out-Null
    $objects.Add("<< /Length $streamLength >>`nstream`n$stream`nendstream") | Out-Null
    $nextObjectNumber += 2
}

$kids = ($pageObjectNumbers | ForEach-Object { "$_ 0 R" }) -join " "
$objects[1] = "<< /Type /Pages /Kids [ $kids ] /Count $($pages.Count) >>"

$builder = New-Object System.Text.StringBuilder
$nl = "`n"
[void]$builder.Append("%PDF-1.4$nl")

$offsets = New-Object System.Collections.Generic.List[int]
for ($index = 0; $index -lt $objects.Count; $index++) {
    $offsets.Add([System.Text.Encoding]::ASCII.GetByteCount($builder.ToString())) | Out-Null
    $objectNumber = $index + 1
    [void]$builder.Append("$objectNumber 0 obj$nl")
    [void]$builder.Append($objects[$index])
    [void]$builder.Append("${nl}endobj${nl}")
}

$xrefStart = [System.Text.Encoding]::ASCII.GetByteCount($builder.ToString())
[void]$builder.Append("xref$nl")
[void]$builder.Append("0 $($objects.Count + 1)$nl")
[void]$builder.Append("0000000000 65535 f $nl")

foreach ($offset in $offsets) {
    [void]$builder.Append(("{0:0000000000} 00000 n $nl" -f $offset))
}

[void]$builder.Append("trailer << /Size $($objects.Count + 1) /Root 1 0 R >>$nl")
[void]$builder.Append("startxref$nl")
[void]$builder.Append("$xrefStart$nl")
[void]$builder.Append("%%EOF")

$outputDir = Split-Path $fullOutputPath -Parent
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

[System.IO.File]::WriteAllBytes($fullOutputPath, [System.Text.Encoding]::ASCII.GetBytes($builder.ToString()))
Write-Output "PDF created at $fullOutputPath"
