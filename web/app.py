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
from datetime import datetime
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

                # Extract ETA if available
                eta = ""
                if "<" in count_part:
                    eta = count_part.split("<")[1].split(",")[0].strip().rstrip("]")

                # Extract description
                desc = s.split(":")[0] if ":" in s.split("%")[0] else ""

                jobs[self.job_id]["progress"] = {
                    "percent": round(pct, 1),
                    "counts": counts,
                    "eta": eta,
                    "description": desc,
                    "raw": s,
                }
        except (ValueError, IndexError):
            pass

        return len(s)

    def flush(self):
        pass


def get_config():
    """Load the default deadwood inference config."""
    config_path = Path(__file__).parent.parent / "deadwood_inference_config.json"
    with open(config_path) as f:
        return json.load(f)


def check_model_available():
    """Check if model weights are downloaded."""
    config = get_config()
    model_path = (
        Path(__file__).parent.parent
        / "data"
        / (config["model_name"] + ".safetensors")
    )
    return model_path.exists()


class CancelledError(Exception):
    pass


def run_inference_job(job_id, filepath, config_overrides):
    """Run deadwood inference in a background thread with progress tracking."""
    try:
        cancel_event = cancel_events.get(job_id)

        def check_cancel():
            if cancel_event and cancel_event.is_set():
                raise CancelledError("Job cancelled by user")

        jobs[job_id]["status"] = "loading_model"
        jobs[job_id]["progress"] = {"percent": 0, "description": "Loading model..."}

        import rasterio
        from common.common import save_poly
        from deadwood.deadwood_inference import DeadwoodInference

        check_cancel()

        # Write temporary config with overrides
        config = get_config()
        config.update(config_overrides)
        tmp_config_path = Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_config.json"
        with open(tmp_config_path, "w") as f:
            json.dump(config, f)

        # Redirect tqdm output to our capture, with cancel checking
        import tqdm

        tqdm_capture = TqdmCapture(job_id)
        original_tqdm_init = tqdm.tqdm.__init__
        original_tqdm_update = tqdm.tqdm.update

        def patched_tqdm_init(self_tqdm, *args, **kwargs):
            kwargs["file"] = tqdm_capture
            kwargs["mininterval"] = 1.0
            original_tqdm_init(self_tqdm, *args, **kwargs)

        def patched_tqdm_update(self_tqdm, n=1):
            check_cancel()
            return original_tqdm_update(self_tqdm, n)

        tqdm.tqdm.__init__ = patched_tqdm_init
        tqdm.tqdm.update = patched_tqdm_update

        try:
            jobs[job_id]["status"] = "running_inference"
            jobs[job_id]["started_at"] = time.time()

            # Get file info for progress context
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            jobs[job_id]["file_info"] = {
                "name": os.path.basename(filepath),
                "size_mb": round(file_size_mb, 1),
            }

            inference = DeadwoodInference(config_path=str(tmp_config_path))
            jobs[job_id]["progress"] = {
                "percent": 0,
                "description": "Starting inference...",
            }
            check_cancel()
            polygons = inference.inference_deadwood(filepath)

            check_cancel()
            jobs[job_id]["status"] = "saving_results"
            jobs[job_id]["progress"] = {
                "percent": 95,
                "description": "Saving results...",
            }

            output_gpkg = (
                Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_deadwood.gpkg"
            )
            crs = rasterio.open(filepath).crs
            save_poly(str(output_gpkg), polygons, crs=crs)

            # Convert polygons to GeoJSON for map display (in EPSG:4326)
            import geopandas as gpd

            gdf = gpd.read_file(str(output_gpkg))
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            geojson_path = (
                Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_deadwood.geojson"
            )
            gdf_wgs84.to_file(str(geojson_path), driver="GeoJSON")

            # Get bounds for map centering
            bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]

            elapsed = time.time() - jobs[job_id].get("started_at", time.time())
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = {"percent": 100, "description": "Complete"}
            jobs[job_id]["result"] = {
                "gpkg_file": f"{job_id}_deadwood.gpkg",
                "geojson_file": f"{job_id}_deadwood.geojson",
                "polygon_count": len(gdf),
                "elapsed_seconds": round(elapsed),
                "bounds": {
                    "south": float(bounds[1]),
                    "west": float(bounds[0]),
                    "north": float(bounds[3]),
                    "east": float(bounds[2]),
                },
            }

        finally:
            # Restore original tqdm
            tqdm.tqdm.__init__ = original_tqdm_init
            tqdm.tqdm.update = original_tqdm_update

        # Clean up temp config
        tmp_config_path.unlink(missing_ok=True)

    except CancelledError:
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["progress"] = {"percent": 0, "description": "Cancelled"}
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
    finally:
        cancel_events.pop(job_id, None)


@app.route("/")
def index():
    model_available = check_model_available()
    config = get_config()
    return render_template("index.html", model_available=model_available, config=config)


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
        "filepath": filepath,
        "started": datetime.now().isoformat(),
        "progress": {"percent": 0, "description": "Queued"},
    }

    thread = Thread(
        target=run_inference_job, args=(job_id, filepath, config_overrides)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


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


@app.route("/outputs/<filename>")
def download_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/geojson/<job_id>")
def get_geojson(job_id):
    geojson_path = Path(app.config["OUTPUT_FOLDER"]) / f"{job_id}_deadwood.geojson"
    if not geojson_path.exists():
        return jsonify({"error": "GeoJSON not found"}), 404
    with open(geojson_path) as f:
        return jsonify(json.load(f))


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
