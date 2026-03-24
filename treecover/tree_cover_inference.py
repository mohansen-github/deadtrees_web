import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from common import (
    filter_polygons_by_area,
    image_reprojector,
    mask_to_polygons,
    reproject_polygons,
)
from tcd_pipeline.pipeline import Pipeline


class TreeCoverInference:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.pipeline = Pipeline(model_or_config="restor/tcd-segformer-mit-b5")

    def inference_treecover(self, input_tif):
        """
        Gets path to tif file and returns polygons of tree cover in the CRS of the tif.
        """

        # reproject to target resolution (TCD expects ~10cm)
        vrt_src = image_reprojector(
            input_tif, min_res=self.config["tree_cover_inference_resolution"]
        )

        # Write VRT to a temp GeoTIFF since TCD Pipeline expects a file path
        tmpdir = tempfile.mkdtemp()
        try:
            temp_tif = os.path.join(tmpdir, "reprojected.tif")
            data = vrt_src.read()
            profile = vrt_src.profile.copy()
            profile.update(driver="GTiff")
            with rasterio.open(temp_tif, "w", **profile) as dst:
                dst.write(data)

            input_crs = vrt_src.crs
            vrt_src.close()

            # run TCD pipeline on the reprojected image
            res = self.pipeline.predict(temp_tif)

            # Read confidence map — TCD returns a DatasetReader, not a numpy array
            confidence_reader = res.confidence_map
            confidence_data = confidence_reader.read(1)

            # threshold the confidence map to binary mask
            threshold = self.config["tree_cover_threshold"]
            outimage = (confidence_data > threshold).astype(np.uint8)

            # convert mask to polygons
            print("Postprocessing mask into polygons and filtering....")
            polygons = mask_to_polygons(outimage, confidence_reader)

            result_crs = confidence_reader.crs

            # Close TCD's handles before cleanup
            confidence_reader.close()
            if hasattr(res, 'image') and res.image is not None:
                res.image.close()

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        # filter small polygons
        polygons = filter_polygons_by_area(
            polygons, self.config["minimum_polygon_area"]
        )

        # reproject polygons back to the CRS of the input tif
        polygons = reproject_polygons(
            polygons, result_crs, rasterio.open(input_tif).crs
        )

        return polygons
