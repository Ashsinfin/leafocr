# LeafOCR

Local palm-leaf manuscript recognition application with a Gradio interface.
This snapshot includes the local UI and pipeline changes built on
[SFR-API](https://github.com/SineeshRemya/SFR-API), original commit `f353e38`.

## Files

- `newds_aug/app.py`: application and OCR pipeline.
- `newds_aug/*.h5` and `*.keras`: segmentation and recognition models.
- `newds_aug/*.ipynb`: project notebooks.
- Sample manuscript images, demo videos, Windows/WSL launchers, and change notes
  are included in `newds_aug`.

## Clone with model files

Install Git LFS before cloning; models and videos are stored in Git LFS.

```sh
git lfs install
git clone https://github.com/Ashsinfin/leafocr.git
cd leafocr
git lfs pull
```

The local Python virtual environment, caches, logs, and machine-specific
Windows shortcut are deliberately excluded. Recreate the Python environment
before launching `newds_aug/app.py` or `run_palm_leaf_ocr.bat`.
The existing WSL setup and GPU launch scripts are also included.
