@echo off
REM Launch LeafOCR on the GPU (inside WSL Ubuntu). Then open http://localhost:7860
wsl -d Ubuntu -- bash /mnt/d/EXPERIMENTS/leaftemp/SFR-API/newds_aug/run_gpu_wsl.sh
pause
