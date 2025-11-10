"""
Optimized Data Loader for SemRoCL Stage 2
Key improvements:
- Better error handling and logging
- Case-insensitive file extensions
- Failure tracking to prevent silent errors
- Improved NaN/Inf validation
"""

import cv2
import os
import glob
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Sized, cast
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Case-insensitive file extensions
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF")


def _list_images(root: str) -> List[str]:
    """Fast image listing using glob with case-insensitive extensions"""
    seen = set()
    paths: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF"):
        for path in glob.glob(os.path.join(root, "**", ext), recursive=True):
            if "__MACOSX" in path:
                continue
            norm_path = os.path.normpath(path)
            base_name = os.path.basename(norm_path)
            if base_name.startswith("._"):
                continue
            key = norm_path.lower()
            if key not in seen:
                seen.add(key)
                paths.append(norm_path)
    return sorted(paths)


class OptimizedLowLightDataset(Dataset):
    """
    Optimized dataset for Stage 2 training with improved error handling
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        paired: Optional[bool] = None,
        img_size: int = 384,
        augment: bool = True,
        normalize: bool = True,
        cache_images: bool = False,
        use_pil: bool = True,
        strict_mode: bool = False,  # NEW: Raise error on invalid data
        return_pair: bool = False,  # NEW: return an additional augmented view
    ):
        super().__init__()
        self.root_dir = os.path.normpath(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.normalize = normalize
        self.cache_images = cache_images
        self.use_pil = use_pil
        self.strict_mode = strict_mode
        self.return_pair = return_pair
        self.image_cache: Optional[Dict[str, np.ndarray]] = {} if cache_images else None
        
        # NEW: Track failed indices to prevent silent failures
        self.failed_indices = set()
        self.max_failures_threshold = 10
        self.nan_count = 0
        
        base_name = os.path.basename(self.root_dir).lower()

        # Auto-detect subdatasets
        if os.path.basename(self.root_dir) == "stage2_paired":
            subdirs = [
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
            if len(subdirs) > 0:
                print(f"Auto-detected subdatasets: {', '.join(os.path.basename(s) for s in subdirs)}")
                self.subdatasets = subdirs
                self.multi_dataset = True
            else:
                self.subdatasets = [self.root_dir]
                self.multi_dataset = False
        else:
            self.subdatasets = [self.root_dir]
            self.multi_dataset = False

        # Special handling for nested datasets (e.g., LSRW split by device)
        if base_name == "lsrw":
            if self.split == "train":
                candidates = ["train", "Training data", "Train"]
            else:
                candidates = ["test", "Eval", "validation", "Validation", "Test"]
            split_path = None
            for name in candidates:
                candidate_path = os.path.join(self.root_dir, name)
                if os.path.isdir(candidate_path):
                    split_path = candidate_path
                    break
            if split_path:
                # If split_path contains multiple device folders (e.g., Huawei/Nikon), treat each separately
                device_dirs = [
                    os.path.join(split_path, d)
                    for d in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, d))
                ]
                # Check if device_dirs actually hold low/high pairs; otherwise, use split_path directly
                valid_device_dirs = []
                for d in device_dirs:
                    has_low = os.path.isdir(os.path.join(d, "low"))
                    has_high = os.path.isdir(os.path.join(d, "high"))
                    if has_low and has_high:
                        valid_device_dirs.append(d)
                if valid_device_dirs:
                    self.subdatasets = valid_device_dirs
                    self.multi_dataset = True
                else:
                    self.subdatasets = [split_path]
                    self.multi_dataset = False
        
        # Collect all image paths
        self.all_low_paths = []
        self.all_high_paths = []
        self.sample_dataset_ids: List[str] = []
        
        for dataset_dir in self.subdatasets:
            low_dir, high_dir, is_paired = self._infer_structure(dataset_dir, paired)

            if not os.path.exists(low_dir):
                print(f"Warning: Low directory not found: {low_dir}")
                continue

            # Scan files
            low_paths = _list_images(low_dir)
            high_paths: List[str] = []

            if is_paired and high_dir and os.path.exists(high_dir):
                high_paths = _list_images(high_dir)
                # Match low and high images by name
                low_names = {os.path.splitext(os.path.basename(p))[0]: p for p in low_paths}
                high_names = {os.path.splitext(os.path.basename(p))[0]: p for p in high_paths}

                # Keep only matching pairs
                common_names = set(low_names.keys()) & set(high_names.keys())
                if not common_names and ("LOL-v2" in dataset_dir or "LOL-v2" in os.path.basename(dataset_dir)):
                    # LOL-v2 uses different prefixes (lowXXXX vs normalXXXX); match by numeric suffix
                    def _normalize_name(name: str) -> str:
                        digits = "".join(ch for ch in name if ch.isdigit())
                        return digits if digits else name

                    low_norm = {_normalize_name(name): path for name, path in low_names.items()}
                    high_norm = {_normalize_name(name): path for name, path in high_names.items()}
                    common_norm = set(low_norm.keys()) & set(high_norm.keys())
                    if common_norm:
                        common_names = sorted(common_norm)
                        low_paths = [low_norm[name] for name in common_names]
                        high_paths = [high_norm[name] for name in common_names]
                    else:
                        low_paths = []
                        high_paths = []
                else:
                    low_paths = [low_names[name] for name in sorted(common_names)]
                    high_paths = [high_names[name] for name in sorted(common_names)]

            self.all_low_paths.extend(low_paths)
            self.all_high_paths.extend(high_paths if high_paths else [None] * len(low_paths))

            relative_path = os.path.relpath(dataset_dir, self.root_dir)
            if relative_path in (".", ""):
                dataset_name = os.path.basename(self.root_dir)
            else:
                dataset_base = os.path.basename(self.root_dir)
                dataset_name = f"{dataset_base}/{relative_path}".replace("\\", "/")
            self.sample_dataset_ids.extend([dataset_name] * len(low_paths))
            print(f"  [{dataset_name}|{self.split}] low={len(low_paths)}, high={len(high_paths) if high_paths else 0}")
        
        self.paired = len(self.all_high_paths) > 0 and self.all_high_paths[0] is not None
        
        if len(self.all_low_paths) == 0:
            print(f"Warning: No images found in {self.root_dir}")
        else:
            print(f"Total samples: {len(self.all_low_paths)}")
        
        # Setup augmentation pipeline
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup augmentation transforms"""
        if self.augment:
            # Gentle augmentation for low-light images
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                # Only geometric augmentations (preserve lighting characteristics)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.3),
                # Normalize to [0, 1]
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2()
            ], additional_targets={'image2': 'image'}, is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2()
            ], additional_targets={'image2': 'image'}, is_check_shapes=False)
    
    def _infer_structure(self, dataset_dir: str, paired_flag: Optional[bool]) -> Tuple[str, Optional[str], bool]:
        """Infer dataset directory structure"""
        rd = dataset_dir
        
        # Check for specific dataset structures
        if "LOL-v1" in rd or "LOL-v1" in os.path.basename(rd):
            low_dir = os.path.join(rd, "low")
            high_dir = os.path.join(rd, "high")
            
            if not os.path.exists(low_dir):
                if self.split == "train":
                    low_dir = os.path.join(rd, "our485", "low")
                    high_dir = os.path.join(rd, "our485", "high")
                else:
                    low_dir = os.path.join(rd, "eval15", "low")
                    high_dir = os.path.join(rd, "eval15", "high")
            
            return low_dir, high_dir, True
        
        elif "LOL-v2" in rd or "LOL-v2" in os.path.basename(rd):
            subset = "Real_captured"
            split_name = "Train" if self.split == "train" else "Test"
            low_dir = os.path.join(rd, subset, split_name, "Low")
            high_dir = os.path.join(rd, subset, split_name, "Normal")
            return low_dir, high_dir, True
        
        elif "FiveK" in rd or "FiveK" in os.path.basename(rd):
            if self.split == "train":
                low_candidates = [
                    os.path.join(rd, "train", "input"),
                    os.path.join(rd, "input")
                ]
                high_candidates = [
                    os.path.join(rd, "train", "expertC"),
                    os.path.join(rd, "train", "target"),
                    os.path.join(rd, "train", "output"),
                    os.path.join(rd, "expertC"),
                    os.path.join(rd, "target"),
                    os.path.join(rd, "output")
                ]
            else:
                low_candidates = [
                    os.path.join(rd, "test", "input"),
                    os.path.join(rd, "input")
                ]
                high_candidates = [
                    os.path.join(rd, "test", "expertC"),
                    os.path.join(rd, "test", "target"),
                    os.path.join(rd, "expertC"),
                    os.path.join(rd, "target")
                ]

            low_dir = next((path for path in low_candidates if os.path.exists(path)), low_candidates[-1])
            high_dir = next((path for path in high_candidates if os.path.exists(path)), None)

            is_paired = high_dir is not None
            return low_dir, high_dir, is_paired
        
        elif "LSRW" in rd or "Real-LOLBlue" in rd:
            low_dir = os.path.join(rd, "low")
            high_dir = os.path.join(rd, "high")
            return low_dir, high_dir, True
        
        # Default structure
        low_dir = os.path.join(rd, "low")
        high_dir = os.path.join(rd, "high")
        
        if not os.path.exists(low_dir):
            if len(_list_images(rd)) > 0:
                return rd, None, False
        
        paired = os.path.isdir(high_dir) if paired_flag is None else paired_flag
        return low_dir, high_dir if paired else None, paired
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and validate image with improved error handling"""
        if path is None:
            raise ValueError("Image path cannot be None")
        
        # Check cache
        if self.cache_images and self.image_cache is not None and path in self.image_cache:
            return self.image_cache[path].copy()
        
        try:
            if self.use_pil:
                img = Image.open(path).convert('RGB')
                img = np.array(img)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to load image: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Validate image dimensions
            if img.shape[2] != 3:
                raise ValueError(f"Invalid image channels: {img.shape}")
            
            # IMPROVED: Better NaN/Inf handling with detailed logging
            if np.isnan(img).any() or np.isinf(img).any():
                nan_count = np.isnan(img).sum()
                inf_count = np.isinf(img).sum()
                print(f"Warning: {os.path.basename(path)} has {nan_count} NaN and {inf_count} Inf values")
                
                self.nan_count += 1
                
                if self.strict_mode:
                    raise ValueError(f"Invalid values in {path}")
                
                # Fix invalid values
                img = np.nan_to_num(img, nan=128.0, posinf=255.0, neginf=0.0)
            
            # Ensure valid range
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Cache if enabled
            if self.cache_images and self.image_cache is not None:
                self.image_cache[path] = img.copy()
            
            return img
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy image as fallback
            return np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
    
    def __len__(self) -> int:
        return len(self.all_low_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get item with improved error handling"""
        try:
            low_path = self.all_low_paths[idx]
            high_path = self.all_high_paths[idx] if idx < len(self.all_high_paths) else None
            
            # Load images
            low_img = self._load_image(low_path)
            high_img = self._load_image(high_path) if high_path else low_img.copy()
            
            # Apply transforms
            if self.paired and high_img is not None:
                transformed = self.transform(image=low_img, image2=high_img)
                low_tensor = transformed['image']
                high_tensor = transformed['image2']
                second_view = None
                if self.return_pair:
                    transformed_pair = self.transform(image=low_img, image2=high_img)
                    second_view = transformed_pair['image']
            else:
                transformed = self.transform(image=low_img)
                low_tensor = transformed['image']
                high_tensor = low_tensor.clone()
                second_view = None
                if self.return_pair:
                    transformed_pair = self.transform(image=low_img)
                    second_view = transformed_pair['image']
            
            # Ensure correct type
            low_tensor = low_tensor.float()
            high_tensor = high_tensor.float()
            if second_view is not None:
                second_view = second_view.float()
            
            # Final safety check (single point, after transform)
            if torch.isnan(low_tensor).any() or torch.isinf(low_tensor).any():
                low_tensor = torch.clamp(torch.nan_to_num(low_tensor, nan=0.5), 0, 1)
            
            if torch.isnan(high_tensor).any() or torch.isinf(high_tensor).any():
                high_tensor = torch.clamp(torch.nan_to_num(high_tensor, nan=0.5), 0, 1)
            
            # Clamp to [0, 1]
            low_tensor = torch.clamp(low_tensor, 0, 1)
            high_tensor = torch.clamp(high_tensor, 0, 1)
            if second_view is not None:
                second_view = torch.clamp(second_view, 0, 1)
            
            sample = {
                'low': low_tensor,
                'high': high_tensor,
                'low_path': low_path
            }
            if second_view is not None:
                sample['low_pair'] = second_view
            
            return sample
            
        except Exception as e:
            # Track failures
            self.failed_indices.add(idx)
            
            if len(self.failed_indices) > self.max_failures_threshold:
                print(f"ERROR: Too many data loading failures ({len(self.failed_indices)})")
                raise RuntimeError(f"Data loading unstable: {len(self.failed_indices)} failures")
            
            print(f"Error in __getitem__ at index {idx}: {e}")
            # Return safe dummy data
            dummy_tensor = torch.ones(3, self.img_size, self.img_size) * 0.5
            return {
                'low': dummy_tensor,
                'high': dummy_tensor.clone(),
                'low_path': 'dummy.jpg'
            }


def get_dataloader(cfg: dict, split="train") -> DataLoader:
    """Create optimized dataloader"""
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    
    # Handle multi-dataset loading
    if "datasets" in ds_cfg and isinstance(ds_cfg["datasets"], list):
        datasets = []
        root_dir = ds_cfg["root_dir"]
        
        for dataset_name in ds_cfg["datasets"]:
            dataset_path = os.path.join(root_dir, dataset_name)
            if os.path.exists(dataset_path):
                ds = OptimizedLowLightDataset(
                    root_dir=dataset_path,
                    split=split,
                    paired=True,
                    img_size=ds_cfg.get("img_size", 384),
                    augment=(split == "train"),
                    normalize=True,
                    cache_images=False,
                    use_pil=True
                )
                if len(ds) > 0:
                    datasets.append(ds)
            else:
                print(f"Warning: Dataset not found: {dataset_path}")
        
        if len(datasets) == 0:
            raise ValueError(f"No valid datasets found in {root_dir}")
        
        dataset = ConcatDataset(datasets)
        print(f"Combined dataset: {len(dataset)} total samples")
    else:
        # Single dataset
        dataset = OptimizedLowLightDataset(
            root_dir=ds_cfg["root_dir"],
            split=split,
            paired=ds_cfg.get("mode", "paired") == "paired",
            img_size=ds_cfg.get("img_size", 384),
            augment=(split == "train"),
            normalize=True,
            cache_images=False,
            use_pil=True
        )

    # Create dataloader
    num_workers = 0 if os.name == 'nt' else train_cfg.get("num_workers", 4)
    sampler = None
    shuffle = (split == "train")

    dataset_counts: Counter[str] = Counter()

    def update_counts(ds_obj):
        ids = getattr(ds_obj, "sample_dataset_ids", None)
        if ids:
            dataset_counts.update(ids)
        else:
            name = getattr(ds_obj, "root_dir", "dataset")
            dataset_counts[name] += len(ds_obj)

    if isinstance(dataset, ConcatDataset):
        for sub_ds in dataset.datasets:
            update_counts(sub_ds)
    else:
        update_counts(dataset)

    def resolve_weight(name: Optional[str]) -> float:
        return 1.0

    if split == "train":
        weight_cfg = ds_cfg.get("sample_weights")
        if weight_cfg:
            def resolve_weight(name: Optional[str]) -> float:
                if not name:
                    return float(weight_cfg.get("default", 1.0))
                key = name.split("/")[-1]
                return float(weight_cfg.get(name, weight_cfg.get(key, weight_cfg.get("default", 1.0))))

            weights: List[float] = []
            if isinstance(dataset, ConcatDataset):
                for sub_ds in dataset.datasets:
                    ids = getattr(sub_ds, "sample_dataset_ids", None)
                    if ids is None:
                        weights.extend([float(weight_cfg.get("default", 1.0))] * len(cast(Sized, sub_ds)))
                    else:
                        weights.extend([resolve_weight(tag) for tag in ids])
            else:
                ids = getattr(dataset, "sample_dataset_ids", None)
                if ids is None:
                    weights = [float(weight_cfg.get("default", 1.0))] * len(cast(Sized, dataset))
                else:
                    weights = [resolve_weight(tag) for tag in ids]

            if len(weights) > 0:
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)  # type: ignore[arg-type]
                shuffle = False

        print("[Stage1] Dataset sampling summary:")
        for name, count in dataset_counts.items():
            print(f"  - {name}: samples={count}, weight={resolve_weight(name):.2f}")

    dataloader_kwargs: Dict[str, Any] = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True) and torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=False if num_workers == 0 else train_cfg.get("persistent_workers", False),
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **dataloader_kwargs,
    )


if __name__ == "__main__":
    print("Testing optimized data loader...")
    
    config = {
        'dataset': {
            'root_dir': './data/stage2_paired',
            'datasets': ['LOL-v1', 'LOL-v2'],
            'img_size': 384,
            'mode': 'paired'
        },
        'training': {
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False
        }
    }
    
    try:
        train_loader = get_dataloader(config, split='train')
        print(f"Dataloader created with {len(train_loader)} batches")
        
        # Test loading
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"  Low: {batch['low'].shape}, range [{batch['low'].min():.2f}, {batch['low'].max():.2f}]")
            print(f"  High: {batch['high'].shape}, range [{batch['high'].min():.2f}, {batch['high'].max():.2f}]")
            
            if i >= 2:
                break
        
        print("\nData loader working correctly!")
    except Exception as e:
        print(f"Error: {e}")


# Compatibility alias for Stage 1
LowLightDataset = OptimizedLowLightDataset
