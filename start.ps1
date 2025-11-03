# Simple Script to Run the Boat Classification Project
Write-Host "Boat Classification Project Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
if (Test-Path "backend\boat_classifier_mobilenet.h5") {
    Write-Host "[OK] Model found" -ForegroundColor Green
} elseif (Test-Path "boat_classifier_mobilenet.h5") {
    Write-Host "[INFO] Moving model to backend folder..." -ForegroundColor Yellow
    Move-Item "boat_classifier_mobilenet.h5" "backend\boat_classifier_mobilenet.h5"
    Write-Host "[OK] Model moved" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Model not found!" -ForegroundColor Red
    Write-Host "Please run all cells in boat-classification.ipynb first" -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Starting backend server..." -ForegroundColor Cyan
Write-Host "Server will run on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

cd backend
python app.py
