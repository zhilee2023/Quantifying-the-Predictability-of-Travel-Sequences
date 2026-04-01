# Requires: Git + Git LFS; origin points to zhilee2023 repo; you have already authenticated (HTTPS PAT or SSH).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== git lfs install ==" -ForegroundColor Cyan
git lfs install

Write-Host "== remote ==" -ForegroundColor Cyan
git remote -v

Write-Host "== status ==" -ForegroundColor Cyan
git status -sb

Write-Host "== push main ==" -ForegroundColor Cyan
git push -u origin main

Write-Host "Done." -ForegroundColor Green
