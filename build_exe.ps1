\
$ErrorActionPreference = "Stop"
Write-Host "== Build NGX_AI_Optimization (Python 3.11) ==" -ForegroundColor Cyan

py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r .\ngx_ai_app\requirements.txt

# TensorFlow (for LSTM). Comment out if you don't need it inside EXE.
pip install tensorflow==2.15.*

pip install pyinstaller

pyinstaller --noconfirm --clean --onedir `
  --name "NGX_AI_Optimization" `
  --add-data "ngx_ai_app\*;ngx_ai_app" `
  --collect-all streamlit `
  --collect-all tensorflow `
  --collect-all keras `
  --collect-all tensorboard `
  desktop_launcher.py

Write-Host "DONE. Run dist\NGX_AI_Optimization\NGX_AI_Optimization.exe" -ForegroundColor Green
