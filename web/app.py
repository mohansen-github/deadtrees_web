"""
Flask web frontend for Deadtree Models - Deadwood Detection in Aerial Imagery.
Provides real-time progress tracking for long-running inference jobs.
"""

import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# Add parent directory to path so we can import the model modules
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(Path(__file__).parent / "uploads")
app.config["OUTPUT_FOLDER"] = str(Path(__file__).parent / "outputs")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB max upload

# Track running jobs and their cancel events
jobs = {}
cancel_events = {}


def _parse_tqdm_time(time_str):
    """Parse tqdm time format (e.g., '02:00', '1:23:45') to total seconds."""
    try:
        parts = time_str.strip().split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            return int(parts[0] * 60 + parts[1])
        elif len(parts) == 3:
            return int(parts[0] * 3600 + parts[1] * 60 + parts[2])
    except (ValueError, IndexError):
        pass
    return None


class TqdmCapture(io.StringIO):
    """Captures tqdm output and updates job progress in real time."""

    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def write(self, s):
        super().write(s)
        s = s.strip()
        if not s or self.job_id not in jobs:
            return len(s)

        # Parse tqdm-style output: "description:  45%|███  | 45/100 [01:23<02:00, 0.50it/s]"
        try:
            if "%" in s and "|" in s:
                # Extract percentage
                pct_part = s.split("%")[0].split()[-1]
                pct = float(pct_part)

                # Extract counts (e.g., "45/100")
                count_part = s.split("|")[-1].strip()
                counts = count_part.split("[")[0].strip()

                # Extract ETA if available and convert to wall-clock time
                eta = ""
                eta_clock = ""
                if "<" in count_part:
                    eta = count_part.split("<")[1].split(",")[0].strip().rstrip("]")
                    # Parse tqdm time format (e.g., "02:00", "1:23:45") to seconds
                    eta_seconds = _parse_tqdm_time(eta)
                    if eta_seconds is not None:
                        finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                        eta_clock = finish_time.strftime("%I:%M %p").lstrip("0")

                # Extract description
                desc = s.split(":")[0] if ":" in s.split("%")[0] else ""

                jobs[self.job_id]["progress"] = {
                    "percent": round(pct, 1),
                    "counts": counts,
                    "eta": eta,
                    "eta_clock": eta_clock,
                    "description": desc,
                    "raw": s,
                }
        except (ValueError, IndexError):
            pass

        return len(s)

    def flush(self):
        pass


def get_config(model_type="deadwood"):
    """Load inference config for the given model type."""
    if model_type == "treecover":
        config_path = Path(__file__).parent.parent / "treecover_inference_config.json"
    else:
        config_path = Path(__file__).parent.parent / "deadwood_inference_config.json"
    with open(config_path) as f:
        return json.load(f)


def check_model_available():
    """Check if deadwood model weights are downloaded."""
    config = get_config("deadwood")
    model_path = (
        Path(__file__).parent.parent
        / "data"
        / (config["model_name"] + ".safetensors")
    )
    return model_path.exists()


def check_tcd_available():
    """Check if the TCD pipeline package is installed."""
    try:
        import tcd_pipeline  # noqa: F401
        return True
    except ImportError:
        return False


class CancelledError(Exception):
    pass


def _setup_tqdm_capture(job_id):
    """Patch tqdm to capture progress and support cancellation."""
    import tqdm

    cancel_event = cancel_events.get(job_id)

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise CancelledError("Job cancelled by user")

    tqdm_capture = TqdmCapture(job_id)
    original_init = tqdm.tqdm.__init__
    original_update = tqdm.tqdm.update

    def patched_init(self_tqdm, *args, **kwargs):
        kwargs["file"] = tqdm_capture
        kwargs["mininterval"] = 1.0
        original_init(self_tqdm, *args, **kwargs)

    def patched_update(self_tqdm, n=1):
        check_cancel()
        return original_update(self_tqdm, n)

    tqdm.tqdm.__init__ = patched_init
    tqdm.tqdm.update = patched_update

    def restore():
        tqdm.tqdm.__init__ = original_init
        tqdm.tqdm.update = original_update

    return check_cancel, restore


def _save_and_finalize(job_id, filepath, polygons, layer_name):
    """Save polygons to gpkg/geojson and update job result."""
    import geopandas as gpd
    import rasterio
    from common.common import save_poly

    # Use the input ortho filename as the base for output files
    ortho_stem = Path(filepath).stem
    base_name = f"{ortho_stem}_{layer_name}"

    output_gpkg = Path(app.config["OUTPUT_FOLDER"]) / f"{base_name}.gpkg"
    crs = rasterio.open(filepath).crs
    save_poly(str(output_gpkg), polygons, crs=crs)

    gdf = gpd.read_file(str(output_gpkg))
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    geojson_path = Path(app.config["OUTPUT_FOLDER"]) / f"{base_name}.geojson"
    gdf_wgs84.to_file(str(geojson_path), driver="GeoJSON")

    bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]

    return {
        "gpkg_file": f"{base_name}.gpkg",
        "geojson_file": f"{base_name}.geojson",
        "polygon_count": len(gdf),
        "bounds": {
            "south": float(bounds[1]),
            "west": float(bounds[0]),
            "north": float(bounds[3]),
            "east": float(bounds[2]),
        },
    }


def run_inference_job(job_id, filepath, config_overrides, model_type="deadwood"):
    """Run inference in a background thread with progress tracking."""
    try:
        jobs[job_id]["status"] = "loading_model"
        jobs[job_id]["progress"] = {"percent": 0, "description": "Loading model..."}

        check_cancel, restore_tqdm = _setup_tqdm_capture(job_id)
        check_cancel()

        # Write temporary config with overrides
        config = get_config(model_type)
        config.update(config_overrides)
        tmp_config_path = Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_config.json"
        with open(tmp_config_path, "w") as f:
            json.dump(config, f)

        try:
            jobs[job_id]["status"] = "running_inference"
            jobs[job_id]["started_at"] = time.time()

            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            jobs[job_id]["file_info"] = {
                "name": os.path.basename(filepath),
                "size_mb": round(file_size_mb, 1),
            }

            jobs[job_id]["progress"] = {
                "percent": 0,
                "description": "Starting inference...",
            }

            if model_type == "treecover":
                from treecover.tree_cover_inference import TreeCoverInference

                inference = TreeCoverInference(config_path=str(tmp_config_path))
                check_cancel()
                polygons = inference.inference_treecover(filepath)
                layer_name = "treecover"
            else:
                from deadwood.deadwood_inference import DeadwoodInference

                inference = DeadwoodInference(config_path=str(tmp_config_path))
                check_cancel()
                polygons = inference.inference_deadwood(filepath)
                layer_name = "deadwood"

            check_cancel()
            jobs[job_id]["status"] = "saving_results"
            jobs[job_id]["progress"] = {
                "percent": 95,
                "description": "Saving results...",
            }

            result = _save_and_finalize(job_id, filepath, polygons, layer_name)

            elapsed = time.time() - jobs[job_id].get("started_at", time.time())
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = {"percent": 100, "description": "Complete"}
            jobs[job_id]["result"] = {
                **result,
                "model_type": model_type,
                "elapsed_seconds": round(elapsed),
            }

        finally:
            restore_tqdm()
            # Release cached GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    except CancelledError:
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["progress"] = {"percent": 0, "description": "Cancelled"}
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
    finally:
        cancel_events.pop(job_id, None)
        # Clean up temp config regardless of outcome
        try:
            tmp_config_path = Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_config.json"
            tmp_config_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/")
def index():
    model_available = check_model_available()
    tcd_available = check_tcd_available()
    config = get_config("deadwood")
    return render_template(
        "index.html",
        model_available=model_available,
        tcd_available=tcd_available,
        config=config,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith((".tif", ".tiff")):
        return (
            jsonify({"error": "Only GeoTIFF files (.tif, .tiff) are supported"}),
            400,
        )

    # Save uploaded file
    safe_name = file.filename.replace(" ", "_")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(filepath)

    return jsonify({"filename": safe_name, "path": filepath})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.json
    filepath = data.get("filepath")

    if not filepath:
        return jsonify({"error": "No filepath provided"}), 400

    # Check if using uploaded file or local path
    if not os.path.isabs(filepath):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filepath)

    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 404

    model_type = data.get("model_type", "deadwood")
    if model_type not in ("deadwood", "treecover"):
        return jsonify({"error": "Invalid model_type"}), 400

    # Config overrides from the UI
    config_overrides = {}
    if "probabilty_threshold" in data:
        config_overrides["probabilty_threshold"] = float(data["probabilty_threshold"])
    if "minimum_polygon_area" in data:
        config_overrides["minimum_polygon_area"] = float(
            data["minimum_polygon_area"]
        )
    if "batch_size" in data:
        config_overrides["batch_size"] = int(data["batch_size"])

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cancel_events[job_id] = Event()
    jobs[job_id] = {
        "status": "queued",
        "model_type": model_type,
        "filepath": filepath,
        "started": datetime.now().isoformat(),
        "progress": {"percent": 0, "description": "Queued"},
    }

    thread = Thread(
        target=run_inference_job,
        args=(job_id, filepath, config_overrides, model_type),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "model_type": model_type})


@app.route("/job/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    if jobs[job_id]["status"] in ("complete", "error", "cancelled"):
        return jsonify({"error": "Job already finished"}), 400
    if job_id in cancel_events:
        cancel_events[job_id].set()
    return jsonify({"status": "cancelling"})


@app.route("/system-stats")
def system_stats():
    """Return current CPU, memory, and GPU utilization."""
    import psutil

    stats = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        },
        "gpu": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": round(mem_allocated, 1),
                "memory_total_gb": round(mem_total, 1),
                "memory_percent": round(mem_allocated / mem_total * 100, 1) if mem_total > 0 else 0,
                "utilization": None,
                "temp_c": None,
            }
            # Get GPU utilization, memory, and temperature via nvidia-smi
            try:
                import subprocess

                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    stats["gpu"]["utilization"] = int(parts[0])
                    mem_used_mb = float(parts[1])
                    mem_total_mb = float(parts[2])
                    stats["gpu"]["memory_used_gb"] = round(mem_used_mb / 1024, 1)
                    stats["gpu"]["memory_total_gb"] = round(mem_total_mb / 1024, 1)
                    stats["gpu"]["memory_percent"] = round(mem_used_mb / mem_total_mb * 100, 1) if mem_total_mb > 0 else 0
                    stats["gpu"]["temp_c"] = int(parts[3])
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
                pass
    except ImportError:
        pass

    return jsonify(stats)


@app.route("/outputs/<filename>")
def download_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/geojson/<job_id>")
def get_geojson(job_id):
    # Determine layer name from job data
    layer_name = "deadwood"
    if job_id in jobs and jobs[job_id].get("model_type") == "treecover":
        layer_name = "treecover"
    if job_id in jobs and "result" in jobs[job_id]:
        # Use the actual geojson filename from the result
        geojson_file = jobs[job_id]["result"].get("geojson_file")
        if geojson_file:
            geojson_path = Path(app.config["OUTPUT_FOLDER"]) / geojson_file
            if geojson_path.exists():
                with open(geojson_path) as f:
                    return jsonify(json.load(f))
    # Fallback
    geojson_path = Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_{layer_name}.geojson"
    if not geojson_path.exists():
        return jsonify({"error": "GeoJSON not found"}), 404
    with open(geojson_path) as f:
        return jsonify(json.load(f))


@app.route("/list-outputs")
def list_outputs():
    """List all output files grouped by input orthophoto."""
    output_dir = Path(app.config["OUTPUT_FOLDER"])
    files = {}
    for f in output_dir.iterdir():
        if f.suffix in (".gpkg", ".geojson"):
            # Group by base name (everything before _deadwood or _treecover)
            name = f.stem
            for suffix in ("_deadwood", "_treecover"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    layer = suffix[1:]  # "deadwood" or "treecover"
                    break
            else:
                base = name
                layer = "unknown"

            if base not in files:
                files[base] = {"name": base, "layers": {}}
            if layer not in files[base]["layers"]:
                files[base]["layers"][layer] = {}
            files[base]["layers"][layer][f.suffix[1:]] = {
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            }

    # Sort by name
    result = sorted(files.values(), key=lambda x: x["name"].lower())
    return jsonify(result)


@app.route("/delete-output", methods=["POST"])
def delete_output():
    """Delete output files for a given base name and optional layer."""
    data = request.json
    base_name = data.get("name", "")
    layer = data.get("layer")  # optional: "deadwood" or "treecover", None = all

    if not base_name:
        return jsonify({"error": "No name provided"}), 400

    output_dir = Path(app.config["OUTPUT_FOLDER"])
    deleted = []

    if layer:
        patterns = [f"{base_name}_{layer}.gpkg", f"{base_name}_{layer}.geojson"]
    else:
        patterns = [
            f"{base_name}_deadwood.gpkg",
            f"{base_name}_deadwood.geojson",
            f"{base_name}_treecover.gpkg",
            f"{base_name}_treecover.geojson",
        ]

    for pattern in patterns:
        target = output_dir / pattern
        if target.exists():
            target.unlink()
            deleted.append(pattern)

    return jsonify({"deleted": deleted})


@app.route("/browse", methods=["POST"])
def browse_local():
    """List TIF files and subdirectories in a given directory."""
    data = request.json
    directory = data.get("directory", "")

    # Default to user's home directory
    if not directory or directory == ".":
        directory = str(Path.home())

    # Special case: if "drives" is requested, list Windows drive letters
    if directory == "__drives__":
        import string
        drives = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append({
                    "name": f"{letter}:\\",
                    "path": drive,
                    "is_dir": True,
                    "is_drive": True,
                })
        return jsonify({"files": drives, "current": "__drives__"})

    directory = os.path.normpath(directory)

    if not os.path.isdir(directory):
        return jsonify({"error": f"Directory not found: {directory}"}), 404

    entries = []
    try:
        for entry in os.scandir(directory):
            try:
                if entry.is_file() and entry.name.lower().endswith((".tif", ".tiff")):
                    entries.append(
                        {
                            "name": entry.name,
                            "path": os.path.normpath(entry.path),
                            "size_mb": round(entry.stat().st_size / (1024 * 1024), 1),
                        }
                    )
                elif entry.is_dir() and not entry.name.startswith("."):
                    entries.append(
                        {
                            "name": entry.name + "/",
                            "path": os.path.normpath(entry.path),
                            "is_dir": True,
                        }
                    )
            except (PermissionError, OSError):
                continue
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    # Sort: directories first, then files, both alphabetically
    dirs = sorted([e for e in entries if e.get("is_dir")], key=lambda x: x["name"].lower())
    files = sorted([e for e in entries if not e.get("is_dir")], key=lambda x: x["name"].lower())

    return jsonify({
        "files": dirs + files,
        "current": directory,
        "parent": str(Path(directory).parent) if Path(directory).parent != Path(directory) else "__drives__",
    })


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    print("Starting Deadtree Web UI on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
