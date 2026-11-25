import json

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
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

# With "tile_averaging", predictions from tiles overlapping by half are blended
# instead. The model often disagrees with itself between neighbouring tiles in
# dense shrubs, which otherwise leaves hard tile edges in the result.
AVERAGING_STRIDE = TILE_SIZE // 2
AVERAGING_EDGE = TILE_SIZE // 4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
            return (probabilities[:, TREE_CLASS] * 255).to(torch.uint8)

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

            trees = (self._predict(images).cpu().numpy() > threshold) & valid

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

    def _predict_mask_averaged(self, src):
        """
        Like _predict_mask, but blends overlapping tiles. Works through the raster
        one row of tiles at a time so that only that row is held in memory.
        """
        stride, edge = AVERAGING_STRIDE, AVERAGING_EDGE
        batch_size = self.config.get("batch_size", 2)
        threshold = self.config["tree_cover_threshold"]

        # the raster is padded by `edge` on each side; rows are in raster pixels
        # and cols are in padded pixels
        rows = list(range(-edge, src.height - edge, stride))
        cols = list(range(0, src.width, stride))
        padded_width = cols[-1] + TILE_SIZE

        window = np.hanning(TILE_SIZE + 2)[1:-1]
        weight = torch.tensor(
            np.outer(window, window) + 1e-3, dtype=torch.float32, device=self.device
        )
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        total = torch.zeros((TILE_SIZE, padded_width), device=self.device)
        weights = torch.zeros_like(total)
        mask = np.zeros((src.height, src.width), dtype=np.uint8)

        image = np.zeros((3, TILE_SIZE, padded_width), dtype=np.uint8)
        valid = np.zeros((TILE_SIZE, padded_width), dtype=bool)

        def read_rows(start, stop):
            """Fills rows start..stop of the current strip from the raster."""
            image[:, start:stop] = 0
            valid[start:stop] = False
            top, bottom = max(row + start, 0), min(row + stop, src.height)
            if top >= bottom:
                return
            raster_window = Window(0, top, src.width, bottom - top)
            strip = np.s_[top - row : bottom - row, edge : edge + src.width]
            image[(slice(None), *strip)] = src.read((1, 2, 3), window=raster_window)
            valid[strip] = src.dataset_mask(window=raster_window) == 255

        with tqdm(total=len(rows) * len(cols), desc="tree cover inference") as progress:
            for row in rows:
                if row == rows[0]:
                    read_rows(0, TILE_SIZE)
                else:
                    image[:, :-stride] = image[:, stride:].copy()
                    valid[:-stride] = valid[stride:].copy()
                    read_rows(TILE_SIZE - stride, TILE_SIZE)

                for i in range(0, len(cols), batch_size):
                    batch = cols[i : i + batch_size]
                    # skip tiles that contain no image data
                    tiles = [c for c in batch if valid[:, c : c + TILE_SIZE].any()]
                    if tiles:
                        pixels = np.stack([image[:, :, c : c + TILE_SIZE] for c in tiles])
                        pixels = torch.from_numpy(pixels).to(self.device).float() / 255
                        confidence = self._predict((pixels - mean) / std)
                        for c, tile in zip(tiles, confidence):
                            total[:, c : c + TILE_SIZE] += tile * weight
                            weights[:, c : c + TILE_SIZE] += weight
                    progress.update(len(batch))

                # rows that no later tile overlaps are final
                done = TILE_SIZE if row == rows[-1] else stride
                trees = (total[:done] / weights[:done].clamp(min=1e-6)) > threshold
                trees = trees.cpu().numpy() & valid[:done]
                top, bottom = max(row, 0), min(row + done, src.height)
                mask[top:bottom] = trees[
                    top - row : bottom - row, edge : edge + src.width
                ]

                for buffer in (total, weights):
                    buffer[:-stride] = buffer[stride:].clone()
                    buffer[-stride:] = 0

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
            if self.config.get("tile_averaging", False):
                mask = self._predict_mask_averaged(vrt_src)
            else:
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
