#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTI Volume Downsampling Tool
Downsample and resize NIFTI volumes to a fixed size (128x128x128)
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import argparse
import os


class VolumeDownsampler:
    """Volume downsampling and resizing tool"""
    
    def __init__(self, target_shape=(128, 128, 128)):
        """
        Initialize downsampler
        
        Args:
            target_shape: Target volume shape (x, y, z)
        """
        self.target_shape = target_shape
    
    def load_nifti(self, nifti_file):
        """
        Load NIFTI file
        
        Args:
            nifti_file: Path to NIFTI file
            
        Returns:
            (volume_data, nifti_img)
        """
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"File not found: {nifti_file}")
        
        print(f"Loading file: {nifti_file}")
        
        # Load NIFTI file
        nifti_img = nib.load(nifti_file)
        volume_data = nifti_img.get_fdata()
        
        # If 4D data, use first timepoint
        if len(volume_data.shape) == 4:
            print(f"  Detected 4D data, using first timepoint (total {volume_data.shape[3]} timepoints)")
            volume_data = volume_data[:, :, :, 0]
        
        print(f"  Original shape: {volume_data.shape}")
        print(f"  Data type: {volume_data.dtype}")
        print(f"  Data range: [{volume_data.min():.2f}, {volume_data.max():.2f}]")
        
        return volume_data, nifti_img
    
    def downsample_volume(self, volume, method='linear'):
        """
        Downsample volume to target shape
        
        Args:
            volume: Input volume (x, y, z)
            method: Interpolation method ('linear', 'nearest', 'cubic')
                   - 'linear': Trilinear interpolation (order=1), smooth, good for most cases
                   - 'nearest': Nearest neighbor (order=0), preserves exact values
                   - 'cubic': Tricubic interpolation (order=3), smoother but may overshoot
            
        Returns:
            Downsampled volume
        """
        original_shape = volume.shape
        
        # Calculate zoom factors for each dimension
        zoom_factors = [
            self.target_shape[0] / original_shape[0],
            self.target_shape[1] / original_shape[1],
            self.target_shape[2] / original_shape[2]
        ]
        
        print(f"\nDownsampling:")
        print(f"  Original shape: {original_shape}")
        print(f"  Target shape: {self.target_shape}")
        print(f"  Zoom factors: [{zoom_factors[0]:.4f}, {zoom_factors[1]:.4f}, {zoom_factors[2]:.4f}]")
        print(f"  Method: {method}")
        
        # Map method name to scipy order
        method_order = {
            'nearest': 0,
            'linear': 1,
            'cubic': 3
        }
        
        if method not in method_order:
            raise ValueError(f"Unsupported method: {method}. Choose from {list(method_order.keys())}")
        
        order = method_order[method]
        
        # Perform downsampling using scipy.ndimage.zoom
        downsampled = zoom(volume, zoom_factors, order=order, mode='nearest')
        
        # Ensure exact target shape (in case of rounding errors)
        if downsampled.shape != self.target_shape:
            print(f"  Adjusting shape from {downsampled.shape} to {self.target_shape}")
            # Crop or pad if necessary
            result = np.zeros(self.target_shape, dtype=downsampled.dtype)
            
            # Calculate slicing indices
            slices = []
            for i in range(3):
                start = max(0, (downsampled.shape[i] - self.target_shape[i]) // 2)
                end = start + min(downsampled.shape[i], self.target_shape[i])
                target_start = max(0, (self.target_shape[i] - downsampled.shape[i]) // 2)
                target_end = target_start + min(downsampled.shape[i], self.target_shape[i])
                slices.append((slice(start, end), slice(target_start, target_end)))
            
            # Copy data
            result[slices[0][1], slices[1][1], slices[2][1]] = \
                downsampled[slices[0][0], slices[1][0], slices[2][0]]
            
            downsampled = result
        
        print(f"  Downsampled shape: {downsampled.shape}")
        print(f"  Data range: [{downsampled.min():.2f}, {downsampled.max():.2f}]")
        
        return downsampled
    
    def save_nifti(self, volume, output_file, original_img=None, use_int16=True):
        """
        Save downsampled volume as NIFTI
        
        Args:
            volume: Downsampled volume
            output_file: Output file path
            original_img: Original NIFTI image (for affine matrix)
            use_int16: If True, convert to int16 before saving (default: True)
        """
        print(f"\nSaving to: {output_file}")
        
        # Convert to int16 if requested
        if use_int16:
            print(f"  Converting to int16...")
            print(f"  Original dtype: {volume.dtype}")
            print(f"  Original range: [{volume.min():.2f}, {volume.max():.2f}]")
            
            # Clip to int16 range and convert
            int16_min = np.iinfo(np.int16).min  # -32768
            int16_max = np.iinfo(np.int16).max  # 32767
            
            # Clip values to int16 range
            volume_clipped = np.clip(volume, int16_min, int16_max)
            
            # Convert to int16
            volume = volume_clipped.astype(np.int16)
            
            print(f"  Converted dtype: {volume.dtype}")
            print(f"  Converted range: [{volume.min()}, {volume.max()}]")
            print(f"  Memory size: {volume.nbytes / (1024*1024):.2f} MB")
        
        # Create affine matrix
        if original_img is not None:
            # Scale the original affine matrix
            original_shape = original_img.shape[:3]
            scale_factors = [
                original_shape[0] / self.target_shape[0],
                original_shape[1] / self.target_shape[1],
                original_shape[2] / self.target_shape[2]
            ]
            
            affine = original_img.affine.copy()
            affine[0, 0] *= scale_factors[0]
            affine[1, 1] *= scale_factors[1]
            affine[2, 2] *= scale_factors[2]
        else:
            # Create identity affine matrix
            affine = np.eye(4)
        
        # Create NIFTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Set header info
        header = nifti_img.header
        header.set_xyzt_units('mm', 'sec')
        
        # Save file
        nib.save(nifti_img, output_file)
        
        print(f"  Successfully saved!")
        print(f"  Shape: {volume.shape}")
        print(f"  Data type: {volume.dtype}")
        print(f"  Voxel size: {header.get_zooms()[0]:.3f} x {header.get_zooms()[1]:.3f} x {header.get_zooms()[2]:.3f} mm^3")
    
    def process(self, input_file, output_file, method='linear'):
        """
        Process downsampling from input to output
        
        Args:
            input_file: Input NIFTI file path
            output_file: Output NIFTI file path
            method: Interpolation method
        """
        try:
            # Load input file
            volume, original_img = self.load_nifti(input_file)
            
            # Downsample
            downsampled = self.downsample_volume(volume, method)
            
            # Save output (use int16 format)
            self.save_nifti(downsampled, output_file, original_img, use_int16=True)
            
            print("\nDownsampling completed successfully!")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='NIFTI Volume Downsampling Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default: linear interpolation)
  python downsampling.py -i input.nii.gz -o output_128.nii.gz
  
  # Use nearest neighbor interpolation
  python downsampling.py -i input.nii.gz -o output_128.nii.gz -m nearest
  
  # Use cubic interpolation (smoother)
  python downsampling.py -i input.nii.gz -o output_128.nii.gz -m cubic
  
  # Custom target size
  python downsampling.py -i input.nii.gz -o output.nii.gz -s 64 64 64
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input NIFTI file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output NIFTI file path'
    )
    parser.add_argument(
        '-s', '--size',
        nargs=3,
        type=int,
        default=[128, 128, 128],
        help='Target size (x y z), default: 128 128 128'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['linear', 'nearest', 'cubic'],
        default='linear',
        help='Interpolation method (default: linear)'
    )
    
    args = parser.parse_args()
    
    # Create downsampler
    target_shape = tuple(args.size)
    downsampler = VolumeDownsampler(target_shape)
    
    # Process
    downsampler.process(args.input, args.output, args.method)


if __name__ == "__main__":
    main()