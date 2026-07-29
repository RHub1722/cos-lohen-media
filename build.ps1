<#
.SYNOPSIS
    Сборка звуковой дорожки косплей-выступления.

.DESCRIPTION
    Обёртка над `python src/build.py`. Находит подходящий Python, проверяет
    наличие FFmpeg и передаёт все аргументы дальше без изменений.

.EXAMPLE
    .\build.ps1

.EXAMPLE
    .\build.ps1 --no-normalize --verbose

.EXAMPLE
    .\build.ps1 --scenario scenario/timeline.json
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BuildArgs
)

$ErrorActionPreference = 'Stop'

# Все пути — относительно каталога этого скрипта (корня проекта).
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Resolve-Python {
    foreach ($candidate in @('python', 'python3', 'py')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $command) { continue }

        $versionOutput = & $candidate --version 2>&1
        if ($LASTEXITCODE -ne 0) { continue }

        $match = [regex]::Match([string]$versionOutput, '(\d+)\.(\d+)')
        if (-not $match.Success) { continue }

        $major = [int]$match.Groups[1].Value
        $minor = [int]$match.Groups[2].Value
        if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) {
            return [pscustomobject]@{ Exe = $candidate; Version = "$major.$minor" }
        }

        Write-Warning "$candidate — версия $major.$minor, требуется 3.11 или новее."
    }
    return $null
}

$python = Resolve-Python
if ($null -eq $python) {
    Write-Host 'Не найден Python 3.11 или новее.' -ForegroundColor Red
    Write-Host 'Установите Python с https://www.python.org/downloads/windows/'
    Write-Host 'и включите опцию "Add python.exe to PATH".'
    exit 2
}

if ($null -eq (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host 'FFmpeg не найден в PATH.' -ForegroundColor Red
    Write-Host 'Установите его командой:  winget install Gyan.FFmpeg'
    Write-Host 'затем откройте новое окно PowerShell и проверьте:  ffmpeg -version'
    Write-Host 'Подробнее — README.md, раздел 1.'
    exit 2
}

Write-Host "Python $($python.Version) ($($python.Exe)), FFmpeg найден." -ForegroundColor DarkGray
Write-Host ''

if ($BuildArgs) {
    & $python.Exe 'src/build.py' @BuildArgs
}
else {
    & $python.Exe 'src/build.py'
}

$code = $LASTEXITCODE
Write-Host ''
if ($code -ne 0) {
    Write-Host "Сборка завершилась с кодом $code." -ForegroundColor Red
}
elseif ($BuildArgs -contains '--validate-only') {
    Write-Host 'Проверки пройдены. Рендер не запускался.' -ForegroundColor Green
}
else {
    Write-Host 'Сборка завершена. Результаты — в папке output\.' -ForegroundColor Green
}
exit $code
