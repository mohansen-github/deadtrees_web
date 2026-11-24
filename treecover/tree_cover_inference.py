import json

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from common import (
    filter_polygons_by_area,
    image_reprojector,
    mask_to_polygons,
    reproject_polygons,
)
from deadwood.InferenceDataset import InferenceDataset

# Restor's tree cover model, loaded directly so that tcd_pipeline (and its
# Detectron2 dependency) is not needed. Weights are CC BY-NC 4.0.
MODEL_ID = "restor/tcd-segformer-mit-b5"
MODEL_REVISION = "ab15b51cdad8fa53cf2ecbb6c738c00b524f3e5a"
TREE_CLASS = 1

# tcd_pipeline predicts 1024px tiles overlapping by 256px and crops half the
# overlap from each side before merging.
TILE_SIZE = 1024
TILE_PADDING = 128


class TreeCoverInference:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
            )
            .to(self.device)
            .eval()
        )

    def _predict(self, images):
        """Returns tree confidence scaled to 0-255, as tcd_pipeline stores it."""
        with torch.inference_mode():
            logits = self.model(pixel_values=images.to(self.device)).logits
            probabilities = torch.nn.functional.interpolate(
                logits, size=images.shape[-2:], mode="bilinear", align_corners=False
            ).softmax(dim=1)
            return (probabilities[:, TREE_CLASS] * 255).to(torch.uint8).cpu().numpy()

    def _predict_mask(self, dataset):
        """Runs tiled inference and returns a binary mask of the whole raster."""
        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 2),
            num_workers=0,
            shuffle=False,
        )
        threshold = self.config["tree_cover_threshold"]
        mask = np.zeros((dataset.height, dataset.width), dtype=np.uint8)

        for nodata_mask, images, windows in tqdm(loader, desc="tree cover inference"):
            valid = nodata_mask.numpy()
            # skip batches that contain no image data
            if not valid.any():
                continue

            trees = (self._predict(images) > threshold) & valid

            for i in range(trees.shape[0]):
                col_off = int(windows["col_off"][i])
                row_off = int(windows["row_off"][i])
                width = int(windows["width"][i])
                height = int(windows["height"][i])

                # clip the cropped window to the raster
                minx, maxx = max(col_off, 0), min(col_off + width, dataset.width)
                miny, maxy = max(row_off, 0), min(row_off + height, dataset.height)
                if minx >= maxx or miny >= maxy:
                    continue

                # position of that region inside the padded tile
                top = miny - row_off + dataset.padding
                left = minx - col_off + dataset.padding
                mask[miny:maxy, minx:maxx] = trees[
                    i, top : top + (maxy - miny), left : left + (maxx - minx)
                ]

        return mask

    def _vectorize(self, mask, image_src):
        """Converts the mask to polygons, in tiles if it is very large."""
        max_pixels = self.config.get("mask_tiling_threshold", 1_000_000_000)
        if mask.size <= max_pixels:
            return mask_to_polygons(mask, image_src)

        step = int(np.sqrt(max_pixels))
        offsets = [
            (y, x)
            for y in range(0, mask.shape[0], step)
            for x in range(0, mask.shape[1], step)
        ]
        polygons = []
        for y, x in tqdm(offsets, desc="vectorizing mask"):
            polygons.extend(
                mask_to_polygons(
                    mask[y : y + step, x : x + step],
                    image_src,
                    offset_x=x,
                    offset_y=y,
                )
            )
        return polygons

    def inference_treecover(self, input_tif):
        """
        Gets path to tif file and returns polygons of tree cover in the CRS of the tif.
        """

        # the model is trained at a fixed resolution (10cm)
        resolution = self.config["tree_cover_inference_resolution"]
        vrt_src = image_reprojector(input_tif, min_res=resolution, max_res=resolution)

        try:
            dataset = InferenceDataset(
                image_src=vrt_src, tile_size=TILE_SIZE, padding=TILE_PADDING
            )
            mask = self._predict_mask(dataset)

            print("Postprocessing mask into polygons and filtering....")
            polygons = self._vectorize(mask, vrt_src)
            mask_crs = vrt_src.crs
        finally:
            vrt_src.close()

        polygons = filter_polygons_by_area(
            polygons, self.config["minimum_polygon_area"]
        )

        # reproject polygons back to the CRS of the input tif
        with rasterio.open(input_tif) as src:
            return reproject_polygons(polygons, mask_crs, src.crs)
