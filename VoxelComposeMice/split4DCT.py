#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split 4D CT volumes where the 4th dimension stores different time frames.
Each timeframe will be saved as an individual 3D NIFTI file inside the target folder.
"""

import argparse
import os
from typing import Tuple

import nibabel as nib
import numpy as np


class FourDCTSplitter:
    """Split 4D CT data (x, y, z, t) into multiple 3D volumes."""

    def __init__(self, output_dir: str, prefix: str = "phase", use_int16: bool = False):
        self.output_dir = output_dir
        self.prefix = prefix
        self.use_int16 = use_int16

    def load_nifti(self, nifti_file: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load the 4D CT nifti file."""
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"File not found: {nifti_file}")

        print(f"Loading file: {nifti_file}")
        nifti_img = nib.load(nifti_file)
        volume = nifti_img.get_fdata()

        if volume.ndim != 4:
            raise ValueError(f"Expected 4D data, got shape {volume.shape}")

        print(f"  Shape: {volume.shape} (x, y, z, time)")
        print(f"  Dtype: {volume.dtype}")
        print(f"  Range: [{volume.min():.2f}, {volume.max():.2f}]")
        return volume, nifti_img

    def ensure_output_dir(self):
        """Create output directory if missing."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")

    def convert_dtype(self, volume: np.ndarray) -> np.ndarray:
        """Optionally cast to int16 for disk savings."""
        if not self.use_int16:
            return volume

        print("  Converting to int16")
        return np.clip(
            volume, np.iinfo(np.int16).min, np.iinfo(np.int16).max
        ).astype(np.int16)

    def split_and_save(self, input_file: str):
        """Split every time frame and save to disk."""
        self.ensure_output_dir()
        volume, nifti_img = self.load_nifti(input_file)

        timepoints = volume.shape[3]
        pad_len = len(str(timepoints))
        print(f"\nSplitting {timepoints} time frames...")

        for idx in range(timepoints):
            frame = volume[..., idx]
            frame = self.convert_dtype(frame)

            header = nifti_img.header.copy()
            header.set_data_shape(frame.shape)
            output_name = f"{self.prefix}_{idx:0{pad_len}d}.nii.gz"
            output_path = os.path.join(self.output_dir, output_name)

            nib.save(nib.Nifti1Image(frame, nifti_img.affine, header), output_path)
            print(f"  Saved frame {idx+1}/{timepoints} -> {output_path}")

        print("\nAll time frames saved successfully!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a 4D CT NIFTI file into multiple 3D volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split4DCT.py -i input_4dct.nii.gz -o output_dir
  python split4DCT.py -i input.nii.gz -o output_dir -p time -t
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input 4D CT NIFTI file")
    parser.add_argument("-o", "--output", required=True, help="Directory to store split volumes")
    parser.add_argument(
        "-p", "--prefix", default="phase", help="Filename prefix for each time frame (default: phase)"
    )
    parser.add_argument(
        "-t",
        "--int16",
        action="store_true",
        help="Convert frames to int16 before saving (saves space)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    splitter = FourDCTSplitter(args.output, args.prefix, args.int16)
    splitter.split_and_save(args.input)


if __name__ == "__main__":
    main()
