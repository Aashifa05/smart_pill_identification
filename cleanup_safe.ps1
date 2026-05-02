<#
Safe cleanup script for this project.
USAGE (dry-run):
  .\cleanup_safe.ps1 -WhatIf
To perform deletion interactively:
  .\cleanup_safe.ps1
To move training CSVs to training_data (interactive confirmation):
  .\cleanup_safe.ps1 -MoveTraining

THIS SCRIPT WILL NOT DELETE PROTECTED FILES OR THE PRODUCTION MODEL.
It performs a dry-run and requires interactive confirmation to delete.
#>

param(
    [switch]$MoveTraining
)

$base = (Get-Location).Path
$outFile = Join-Path $base "cleanup_report.txt"
if (Test-Path $outFile) { Remove-Item $outFile -Force }

# Define protected paths (relative) - do not remove these
$protected = @(
    "Detection_and_Analysis_of_Pill\models\production\model_feature_learning_mobilenetv3.keras",
    "Detection_and_Analysis_of_Pill\models\production\metadata.json",
    "Detection_and_Analysis_of_Pill\models\production\version.txt",
    "Detection_and_Analysis_of_Pill\static\plots\accuracy_plot_final.png",
    "Detection_and_Analysis_of_Pill\static\plots\confusion_matrix_final.png",
    "Detection_and_Analysis_of_Pill\static\plots\loss_plot_final.png",
    "media\pill_predictions.csv",
    "media\pilldata\class_mapping.csv"
)

function IsProtected($f) {
    foreach ($p in $protected) {
        try {
            $pf = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
n            if ($pf -and ($f -like ("*$($pf.Path)"))) { return $true }
        } catch { }
    }
    return $false
}

# Gather candidates
$modelsOut = Get-ChildItem -Path . -Include *.keras,*.h5 -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "\\models\\production\\" }
$checkpoints = Get-ChildItem -Path . -Include *.ckpt,*.index,*.data-* -Recurse -File -ErrorAction SilentlyContinue
$pycacheDirs = Get-ChildItem -Path . -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue
$pycFiles = Get-ChildItem -Path . -Include *.pyc -Recurse -File -ErrorAction SilentlyContinue
$dsstore = Get-ChildItem -Path . -Include .DS_Store -Recurse -File -ErrorAction SilentlyContinue
$logs = Get-ChildItem -Path . -Include *.log,*.tmp -Recurse -File -ErrorAction SilentlyContinue

# CSVs that may be moved or kept
$csvs = Get-ChildItem -Path . -Include *.csv -Recurse -File -ErrorAction SilentlyContinue

Add-Content $outFile "CLEANUP DRY-RUN - $(Get-Date)"
Add-Content $outFile "\n-- Models (candidates outside production) --"
$modelsOut | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- Checkpoints/TF artifacts --"
$checkpoints | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- __pycache__ directories --"
$pycacheDirs | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- .pyc files --"
$pycFiles | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- .DS_Store files --"
$dsstore | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- Logs/temps --"
$logs | ForEach-Object { Add-Content $outFile $_.FullName }
Add-Content $outFile "\n-- CSV files found --"
$csvs | ForEach-Object { Add-Content $outFile $_.FullName }

Write-Host "Dry-run report written to $outFile" -ForegroundColor Green
Write-Host "Review the file before proceeding. No deletions performed yet." -ForegroundColor Yellow

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "Models (candidates): $($modelsOut.Count)" -ForegroundColor Yellow
Write-Host "Checkpoints: $($checkpoints.Count)" -ForegroundColor Yellow
Write-Host "__pycache__ dirs: $($pycacheDirs.Count)" -ForegroundColor Yellow
Write-Host ".pyc files: $($pycFiles.Count)" -ForegroundColor Yellow
Write-Host "Logs/temps: $($logs.Count)" -ForegroundColor Yellow

# Ask for confirmation
$confirm = Read-Host "Type YES to delete the listed candidate files (irreversible)"
if ($confirm -ne 'YES') { Write-Host "Aborting deletion. No files changed."; exit 0 }

# Deletion - double-check protection
Write-Host "Deleting candidates now..." -ForegroundColor Red

foreach ($f in $modelsOut) {
    if (-not (IsProtected $f.FullName)) { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue; Write-Host "Deleted: $($f.FullName)" }
    else { Write-Host "Skipped protected model: $($f.FullName)" -ForegroundColor Green }
}
foreach ($f in $checkpoints) {
    if (-not (IsProtected $f.FullName)) { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue; Write-Host "Deleted: $($f.FullName)" }
}
foreach ($d in $pycacheDirs) {
    Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "Removed dir: $($d.FullName)" }
foreach ($f in $pycFiles) { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue; Write-Host "Deleted: $($f.FullName)" }
foreach ($f in $dsstore) { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue; Write-Host "Deleted: $($f.FullName)" }
foreach ($f in $logs) { Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue; Write-Host "Deleted: $($f.FullName)" }

# Optionally move training CSVs to training_data/
if ($MoveTraining) {
    $trainingDir = Join-Path $base 'training_data'
    if (-not (Test-Path $trainingDir)) { New-Item -Path $trainingDir -ItemType Directory | Out-Null }
    $toMove = $csvs | Where-Object { $_.Name -match 'Training_set|Testing_set' }
    foreach ($m in $toMove) {
        $dest = Join-Path $trainingDir $m.Name
        Move-Item -LiteralPath $m.FullName -Destination $dest -Force
        Write-Host "Moved $($m.FullName) -> $dest"
    }
}

Write-Host "Cleanup complete. Re-run Django system check if needed." -ForegroundColor Green
