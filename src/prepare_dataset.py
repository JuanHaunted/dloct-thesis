"""
Dataset Preparation Script for Lateral Super-Resolution

This script prepares training pairs for physics-informed deep learning
super-resolution of laterally subsampled OCT tomograms.

Process:
1. Load tomograms from data/train/synthetic and data/train/phase
2. Analyze the Mean Power Spectrum (MPS) to determine sampling state
3. Compute the critical subsampling factor for each tomogram
4. Create subsampled versions at various factors
5. Save training pairs (subsampled, original) to data/processed

Usage:
    python -m src.prepare_dataset --target-hwhm 0.6 --output-dir data/processed

Author: DLOCT Thesis Project
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from .lateral_sampling import (
    determine_subsampling_factor,
    analyze_subsampling,
    create_training_pair,
    fit_gaussian_to_mps,
    compute_mps_1d
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TomogramMetadata:
    """Metadata for a processed tomogram."""
    source_file: str
    source_type: str  # 'synthetic' or 'phase'
    shape: tuple
    dtype: str
    original_hwhm_fast: float
    original_hwhm_slow: float
    critical_factor_fast: float
    critical_factor_slow: float
    is_oversampled_fast: bool
    is_oversampled_slow: bool


@dataclass
class TrainingPairMetadata:
    """Metadata for a training pair."""
    source_file: str
    subsampling_axis: str  # 'fast' or 'slow'
    subsampling_factor: float
    original_hwhm: float
    apparent_hwhm: float
    is_undersampled: bool


def load_tomogram(filepath: Path) -> np.ndarray:
    """
    Load a tomogram from .npy file with format detection.

    Parameters
    ----------
    filepath : Path
        Path to .npy file

    Returns
    -------
    np.ndarray
        Tomogram array (complex or real/imag channels)
    """
    tom = np.load(filepath)
    logger.info(f"Loaded {filepath.name}: shape={tom.shape}, dtype={tom.dtype}")
    return tom


def to_complex(tom: np.ndarray) -> np.ndarray:
    """Convert tomogram to complex representation."""
    if np.iscomplexobj(tom):
        return tom
    elif tom.ndim == 4 and tom.shape[-1] == 2:
        return tom[..., 0] + 1j * tom[..., 1]
    elif tom.ndim == 4:
        # Polarimetric - use first channel
        return to_complex(tom[..., 0])
    else:
        return tom


def from_complex(tom: np.ndarray) -> np.ndarray:
    """Convert complex tomogram to real/imag representation."""
    if np.iscomplexobj(tom):
        return np.stack([tom.real, tom.imag], axis=-1)
    return tom


def analyze_tomogram(
    tom: np.ndarray,
    source_file: str,
    source_type: str,
    n_depth_samples: int = 10
) -> TomogramMetadata:
    """
    Analyze a tomogram's lateral sampling properties.

    Parameters
    ----------
    tom : np.ndarray
        Input tomogram
    source_file : str
        Source filename
    source_type : str
        'synthetic' or 'phase'
    n_depth_samples : int
        Number of depth slices to average

    Returns
    -------
    TomogramMetadata
        Analysis results
    """
    tom_cx = to_complex(tom)

    # Analyze fast axis (x, axis=1 in 3D)
    analysis_fast = determine_subsampling_factor(
        tom_cx, target_hwhm=0.5, axis=1, n_samples=n_depth_samples
    )

    # Analyze slow axis (y, axis=2 in 3D)
    analysis_slow = determine_subsampling_factor(
        tom_cx, target_hwhm=0.5, axis=2, n_samples=n_depth_samples
    )

    return TomogramMetadata(
        source_file=source_file,
        source_type=source_type,
        shape=tom.shape,
        dtype=str(tom.dtype),
        original_hwhm_fast=float(analysis_fast['original_hwhm']),
        original_hwhm_slow=float(analysis_slow['original_hwhm']),
        critical_factor_fast=float(analysis_fast['recommended_factor']),
        critical_factor_slow=float(analysis_slow['recommended_factor']),
        is_oversampled_fast=bool(analysis_fast['is_oversampled']),
        is_oversampled_slow=bool(analysis_slow['is_oversampled'])
    )


def create_subsampled_dataset(
    tom: np.ndarray,
    metadata: TomogramMetadata,
    target_hwhm: float,
    output_dir: Path,
    axis: str = 'fast',
    factors: Optional[List[float]] = None
) -> List[TrainingPairMetadata]:
    """
    Create subsampled training pairs from a tomogram.

    Parameters
    ----------
    tom : np.ndarray
        Input tomogram
    metadata : TomogramMetadata
        Tomogram analysis results
    target_hwhm : float
        Target HWHM for undersampling (should be > 0.5)
    output_dir : Path
        Directory to save outputs
    axis : str
        'fast' or 'slow'
    factors : list, optional
        Specific subsampling factors to use

    Returns
    -------
    list
        List of TrainingPairMetadata for created pairs
    """
    if axis == 'fast':
        axis_idx = 1
        original_hwhm = metadata.original_hwhm_fast
        critical_factor = metadata.critical_factor_fast
    else:
        axis_idx = 2
        original_hwhm = metadata.original_hwhm_slow
        critical_factor = metadata.critical_factor_slow

    # Determine factors to use
    if factors is None:
        # Create a range from 1.0 up to slightly beyond critical factor
        if critical_factor > 1:
            max_factor = min(critical_factor * 1.5, 4.0)
            factors = np.linspace(1.0, max_factor, 5)
        else:
            # Already at or beyond Nyquist
            factors = [1.0, 1.5, 2.0]

    # Filter to only factors that would create undersampling
    min_factor_for_target = target_hwhm / original_hwhm
    factors = [f for f in factors if f >= min_factor_for_target * 0.9]

    if not factors:
        logger.warning(f"No valid factors for {metadata.source_file} {axis} axis")
        return []

    pairs_metadata = []
    source_stem = Path(metadata.source_file).stem

    for factor in factors:
        apparent_hwhm = factor * original_hwhm

        # Create training pair
        subsampled, original = create_training_pair(tom, factor, axis_idx)

        # Create output filenames
        factor_str = f"{factor:.2f}".replace('.', 'p')
        pair_name = f"{source_stem}_{axis}_f{factor_str}"

        # Save
        subsampled_path = output_dir / f"{pair_name}_subsampled.npy"
        original_path = output_dir / f"{pair_name}_original.npy"

        np.save(subsampled_path, subsampled)
        np.save(original_path, original)

        pair_meta = TrainingPairMetadata(
            source_file=metadata.source_file,
            subsampling_axis=axis,
            subsampling_factor=float(factor),
            original_hwhm=float(original_hwhm),
            apparent_hwhm=float(apparent_hwhm),
            is_undersampled=apparent_hwhm > 0.5
        )
        pairs_metadata.append(pair_meta)

        logger.info(
            f"Created pair: {pair_name} "
            f"(factor={factor:.2f}, HWHM={apparent_hwhm:.3f}, "
            f"undersampled={apparent_hwhm > 0.5})"
        )

    return pairs_metadata


def process_all_tomograms(
    data_dir: Path,
    output_dir: Path,
    target_hwhm: float = 0.6,
    axes: List[str] = ['fast', 'slow']
) -> Dict[str, Any]:
    """
    Process all tomograms in the data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing phase/ and synthetic/ subdirectories
    output_dir : Path
        Directory to save processed data
    target_hwhm : float
        Target HWHM for undersampling
    axes : list
        Which axes to process ('fast', 'slow', or both)

    Returns
    -------
    dict
        Summary of processing including all metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'target_hwhm': target_hwhm,
        'tomograms': [],
        'training_pairs': []
    }

    # Find all tomograms
    source_dirs = [
        ('synthetic', data_dir / 'train' / 'synthetic'),
        ('phase', data_dir / 'train' / 'phase')
    ]

    all_files = []
    for source_type, source_dir in source_dirs:
        if source_dir.exists():
            for f in source_dir.glob('*.npy'):
                all_files.append((source_type, f))

    if not all_files:
        logger.warning(f"No .npy files found in {data_dir}")
        return summary

    logger.info(f"Found {len(all_files)} tomogram files")

    for source_type, filepath in tqdm(all_files, desc="Processing tomograms"):
        try:
            # Load tomogram
            tom = load_tomogram(filepath)

            # Analyze sampling properties
            metadata = analyze_tomogram(tom, filepath.name, source_type)
            summary['tomograms'].append(asdict(metadata))

            logger.info(
                f"\n{filepath.name}:\n"
                f"  Fast axis: HWHM={metadata.original_hwhm_fast:.3f}, "
                f"critical_factor={metadata.critical_factor_fast:.2f}\n"
                f"  Slow axis: HWHM={metadata.original_hwhm_slow:.3f}, "
                f"critical_factor={metadata.critical_factor_slow:.2f}"
            )

            # Create training pairs for each requested axis
            for axis in axes:
                pairs = create_subsampled_dataset(
                    tom, metadata, target_hwhm, output_dir, axis
                )
                summary['training_pairs'].extend([asdict(p) for p in pairs])

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue

    # Save summary
    summary_path = output_dir / 'dataset_metadata.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Tomograms analyzed: {len(summary['tomograms'])}")
    logger.info(f"  Training pairs created: {len(summary['training_pairs'])}")
    logger.info(f"  Metadata saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Prepare lateral super-resolution dataset'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=Path('data'),
        help='Directory containing train/synthetic and train/phase'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('data/processed'),
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--target-hwhm', type=float, default=0.6,
        help='Target HWHM for undersampling (should be > 0.5)'
    )
    parser.add_argument(
        '--axes', nargs='+', choices=['fast', 'slow'],
        default=['fast', 'slow'],
        help='Which lateral axes to process'
    )
    parser.add_argument(
        '--analyze-only', action='store_true',
        help='Only analyze, do not create training pairs'
    )

    args = parser.parse_args()

    if args.target_hwhm <= 0.5:
        logger.warning(
            f"target_hwhm={args.target_hwhm} is at or below Nyquist. "
            f"Consider using > 0.5 for genuine undersampling."
        )

    if args.analyze_only:
        # Just analyze all tomograms
        source_dirs = [
            ('synthetic', args.data_dir / 'train' / 'synthetic'),
            ('phase', args.data_dir / 'train' / 'phase')
        ]

        for source_type, source_dir in source_dirs:
            if not source_dir.exists():
                continue

            for filepath in source_dir.glob('*.npy'):
                tom = load_tomogram(filepath)
                metadata = analyze_tomogram(tom, filepath.name, source_type)

                print(f"\n{filepath.name}:")
                print(f"  Shape: {metadata.shape}")
                print(f"  Fast axis:")
                print(f"    HWHM: {metadata.original_hwhm_fast:.4f}")
                print(f"    Critical factor: {metadata.critical_factor_fast:.2f}")
                print(f"    Oversampled: {metadata.is_oversampled_fast}")
                print(f"  Slow axis:")
                print(f"    HWHM: {metadata.original_hwhm_slow:.4f}")
                print(f"    Critical factor: {metadata.critical_factor_slow:.2f}")
                print(f"    Oversampled: {metadata.is_oversampled_slow}")
    else:
        process_all_tomograms(
            args.data_dir,
            args.output_dir,
            args.target_hwhm,
            args.axes
        )


if __name__ == '__main__':
    main()
