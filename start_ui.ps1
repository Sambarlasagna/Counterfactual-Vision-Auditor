# Start CVA Visual Reliability Lab (Backend + Frontend)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting CVA Visual Reliability Lab" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$backendProcess = Start-Process -FilePath "d:\Programming\Projects\ml-venv\Scripts\python.exe" -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port 8000" -WorkingDirectory "d:\Programming\Projects\Counterfactual-Vision-Auditor\backend" -PassThru
Write-Host "[1/2] FastAPI backend running at: http://127.0.0.1:8000 (PID: $($backendProcess.Id))" -ForegroundColor Green

$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory "d:\Programming\Projects\Counterfactual-Vision-Auditor\frontend" -PassThru
Write-Host "[2/2] Vite React frontend running at: http://localhost:5173 (PID: $($frontendProcess.Id))" -ForegroundColor Green

Write-Host "`nOpen http://localhost:5173 in your browser to launch Studio." -ForegroundColor Yellow
Write-Host "Press Ctrl+C or close this window when done." -ForegroundColor Gray
