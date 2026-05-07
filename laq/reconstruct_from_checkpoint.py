"""
Script to generate reconstructed images from a pretrained LAQ model checkpoint.
"""

import torch
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T

from laq_model import LatentActionQuantization
from laq_model.data import ImageVideoDataset
from torch.utils.data import Dataset, DataLoader, Subset


def exists(val):
    return val is not None


def custom_collate_fn(batch):
    """Custom collate function that center-crops images to fixed size."""
    try:
        target_height, target_width = 256, 256
        
        cropped_batch = []
        for item in batch:
            # item shape: [C, T, H, W] where C=3, T=2
            h, w = item.shape[-2:]
            
            # Center crop
            if h > target_height or w > target_width:
                start_h = max(0, (h - target_height) // 2)
                start_w = max(0, (w - target_width) // 2)
                item = item[..., start_h:start_h+target_height, start_w:start_w+target_width]
            
            # Pad if smaller than target
            if h < target_height or w < target_width:
                pad_h = target_height - item.shape[-2]
                pad_w = target_width - item.shape[-1]
                item = torch.nn.functional.pad(item, (0, pad_w, 0, pad_h))
            
            cropped_batch.append(item)
        
        return torch.stack(cropped_batch)
    except Exception as e:
        print(f"Collate error: {type(e).__name__}: {e}")
        return None


class SafeImageVideoDataset(Dataset):
    """Wrapper around ImageVideoDataset that skips samples that fail to load."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = []
        self._find_valid_indices()
    
    def _find_valid_indices(self):
        """Pre-compute which indices load successfully."""
        print("Finding valid samples...")
        # Increase recursion limit temporarily to handle nested recursion in corrupted samples
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(150)  # Set low limit to fail fast on recursive errors
        
        for idx in tqdm(range(len(self.dataset)), desc="Validating samples"):
            try:
                _ = self.dataset[idx]
                self.valid_indices.append(idx)
            except (RecursionError, Exception) as e:
                # Skip samples that fail to load
                pass
        
        sys.setrecursionlimit(old_limit)  # Restore original limit
        print(f"Found {len(self.valid_indices)} valid samples out of {len(self.dataset)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.dataset[actual_idx]


def main():
    parser = argparse.ArgumentParser(description='Generate reconstructions from pretrained LAQ model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the LAQ model checkpoint')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to folder containing image/video data')
    parser.add_argument('--output_dir', type=str, default='./reconstructions', help='Output directory for reconstructed images')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (default: 256)')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size (default: 32)')
    parser.add_argument('--codebook_size', type=int, default=8, help='Codebook size (default: 8)')
    parser.add_argument('--dim', type=int, default=1024, help='Model dimension (default: 1024)')
    parser.add_argument('--quant_dim', type=int, default=32, help='Quantization dimension (default: 32)')
    parser.add_argument('--heads', type=int, default=16, help='Number of attention heads (default: 16)')
    parser.add_argument('--dim_head', type=int, default=64, help='Dimension per head (default: 64)')
    parser.add_argument('--spatial_depth', type=int, default=8, help='Spatial transformer depth (default: 8)')
    parser.add_argument('--temporal_depth', type=int, default=8, help='Temporal transformer depth (default: 8)')
    parser.add_argument('--code_seq_len', type=int, default=4, help='Code sequence length (default: 4)')
    parser.add_argument('--offset', type=int, default=5, help='Frame offset for temporal sampling (default: 5)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of image/video sequences to process (default: all)')
    parser.add_argument('--train_on_images', action='store_true', help='If training was on images instead of videos')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation step (load samples on-the-fly with error handling)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Random seed set to: {args.seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir).joinpath(f"laq_reconstructions_{Path(args.checkpoint).stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing LAQ model...")
    model = LatentActionQuantization(
        dim=args.dim,
        quant_dim=args.quant_dim,
        codebook_size=args.codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        code_seq_len=args.code_seq_len,
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Handle both full state dict and wrapped model state dicts
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # If the checkpoint has a 'model' key, it's from the trainer
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_folder}")
    base_dataset = ImageVideoDataset(
        args.data_folder,
        image_size=args.image_size,
        offset=args.offset
    )
    
    # Wrap with safety layer to skip corrupted samples (unless skipped)
    if not args.skip_validation:
        print("Validating dataset samples (this may take a while)...")
        dataset = SafeImageVideoDataset(base_dataset)
        if len(dataset) == 0:
            print("WARNING: Validation found 0 valid samples!")
            print("Try using --skip_validation flag to load samples on-the-fly instead")
    else:
        print("Skipping validation, will attempt to load samples during processing...")
        dataset = base_dataset
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        num_to_process = min(args.num_samples, len(dataset))
        dataset = Subset(dataset, list(range(num_to_process)))
        print(f"Limiting to {num_to_process} sequences")
    
    # Create dataloader with num_workers=0 to avoid multiprocessing issues
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Generate reconstructions
    print("Generating reconstructions...")
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, total=num_batches)):
            # Skip if collate_fn returned None (collation failed)
            if data is None:
                print(f"Skipping batch {batch_idx}: collation failed")
                continue
                
            try:
                data = data.to(device)
                
                # Get reconstructions
                recons = model(data, return_recons_only=True)
                
                if args.train_on_images:
                    # For image data
                    imgs_and_recons = torch.stack((data, recons), dim=0)
                    from einops import rearrange
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow=2, normalize=True, value_range=(0, 1))
                else:
                    # For video data (showing first frame, last frame, and reconstruction)
                    imgs_and_recons = torch.stack((data[:, :, 0], data[:, :, -1], recons), dim=0)
                    from einops import rearrange
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow=3, normalize=True, value_range=(0, 1))
                
                # Save grid image
                output_path = output_dir / f'reconstruction_batch_{batch_idx:04d}.png'
                save_image(grid, str(output_path))
            except Exception as e:
                print(f"Skipping batch {batch_idx}: {type(e).__name__} - {str(e)[:100]}")
    
    print(f"\nReconstruction complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
