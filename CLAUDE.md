# Deadtree Models

Deadwood detection in aerial/drone orthophoto imagery using deep learning semantic segmentation.
Author: Clemens Mosig (Universität Leipzig). MIT License.

## Project Structure

```
Deadtree/
├── common/                  # Shared GIS/raster utilities
│   ├── __init__.py          # Re-exports everything from common.py
│   └── common.py            # image_reprojector, mask_to_polygons, save_poly, reproject_polygons, filter_polygons_by_area, merge_polygons
├── deadwood/                # Main deadwood detection module
│   ├── __init__.py          # Re-exports DeadwoodInference
│   ├── deadwood_inference.py  # DeadwoodInference class - loads model, runs tiled inference, returns polygons
│   └── InferenceDataset.py  # PyTorch Dataset for tiled reading of large rasters
├── treecover/               # Tree cover detection (incomplete/not running)
│   └── tree_cover_inference.py
├── web/                     # Flask web frontend (added locally, not upstream)
│   ├── app.py               # Flask app with upload, inference, progress tracking, map display
│   ├── templates/index.html # Dark-themed UI with Leaflet map
│   ├── uploads/             # Uploaded orthophoto files
│   └── outputs/             # Inference output files (gpkg, geojson)
├── data/                    # Model weights directory (gitignored)
│   └── segformer_b5_full_epoch_100.safetensors  # Download from Google Drive
├── infer_ortho.py           # Single-file inference script
├── infer_ortho_batch.py     # Batch inference from file list
├── tileinference.py         # Standalone tile inference script (legacy)
├── deadwood_inference_config.json
├── treecover_inference_config.json
└── requirements.txt
```

## Key Concepts

- **Model**: SegFormer B5 encoder with UNet decoder (via segmentation-models-pytorch), binary segmentation (deadwood vs background)
- **Input**: GeoTIFF orthophotos (aerial/drone RGB imagery)
- **Output**: GeoPackage (.gpkg) files containing polygons of detected deadwood areas
- **Pipeline**: Load TIF → reproject to UTM at ≥0.05m resolution → tile into 1024x1024 patches with 256px padding → run model → threshold probabilities → vectorize mask → filter small polygons → reproject back to original CRS
- **Normalization**: ImageNet mean/std (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Config note**: The threshold field is misspelled as `probabilty_threshold` (missing 'i') — this is intentional, do not fix it as it would break config loading

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web frontend
pip install flask
python -m web.app
# Opens at http://localhost:5000

# Run CLI inference (single file)
python infer_ortho.py

# Run CLI batch inference
python infer_ortho_batch.py path_list.txt
```

## Dependencies

Python 3. Key packages: torch, torchvision, segmentation-models-pytorch, rasterio, geopandas, opencv-python-headless, shapely, safetensors, numpy, tqdm, utm. GPU (CUDA) strongly recommended — inference on CPU is very slow.

## Model Weights

Not included in repo. Download from: https://drive.google.com/file/d/1ZvUkBNVtXFujHo_amGyhSXWnPiAc_VES/view
Place the `.safetensors` file in `data/` directory.

## Notes

- `treecover/` module is incomplete — references undefined functions (`reproject_to_10cm`, `Pipeline`) and has bugs (`temp` vs `tempdir`). Not operational.
- `tileinference.py` is a standalone legacy script with hardcoded paths. The `deadwood/` module is the maintained inference path.
- Large orthophotos can take hours to process. The web UI tracks progress via tqdm capture.
- Mask-to-polygon conversion uses OpenCV `findContours` with `RETR_CCOMP` for proper hole handling.
- For very large masks (>1B pixels), vectorization is done in tiles to avoid memory issues.
