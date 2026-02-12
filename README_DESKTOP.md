# NGX AI Optimization — Desktop Rebuild v3

This rebuild removes the auto Chapter 4 preview UI and keeps:
- Multi-sheet Excel upload (each sheet = a sector/index)
- Regime-conditioned portfolio backtest
- Charts + tables
- Plain-English explanation files + sector-by-sector explanation

## Test as normal Streamlit first
```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\activate
pip install -r .\ngx_ai_app\requirements.txt
pip install tensorflow==2.15.*
streamlit run .\ngx_ai_app\app.py
```

## Build EXE (TensorFlow bundled)
PowerShell:
```powershell
.\build_exe.ps1
```

CMD:
```bat
build_exe.bat
```

## Run
Run the EXE from its folder (onedir):
`dist\NGX_AI_Optimization\NGX_AI_Optimization.exe`

If it fails, open:
`dist\NGX_AI_Optimization\launcher.log`
