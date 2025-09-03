"""
Bi-temporal Dataset for Wildfire Burnt Area Detection

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class BiTemporalDataset(Dataset):
    """
    Bi-temporal dataset for wildfire burnt area detection
    
    Args:
        pre_dir (str): Directory containing pre-fire images
        post_dir (str): Directory containing post-fire images
        mask_dir (str): Directory containing burnt area masks
        transform (albumentations.Compose, optional): Data augmentation pipeline
        image_size (tuple): Target image size (height, width)
        normalize (bool): Whether to apply ImageNet normalization
    """
    
    def __init__(self, pre_dir, post_dir, mask_dir=None, transform=None, 
                 image_size=(224, 224), normalize=True):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # Default transform
        if transform is None:
            transforms_list = [A.Resize(image_size[0], image_size[1])]
            if normalize:
                transforms_list.append(A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ))
            transforms_list.append(ToTensorV2())
            self.transform = A.Compose(transforms_list)
        else:
            self.transform = transform
        
        # Find matching image pairs
        self._build_dataset()
    
    def _build_dataset(self):
        """Build dataset by finding matching pre_fire/post_fire image pairs"""
        self.samples = []
        
        if not os.path.exists(self.pre_dir):
            raise ValueError(f"Pre-event directory not found: {self.pre_dir}")
        if not os.path.exists(self.post_dir):
            raise ValueError(f"Post-event directory not found: {self.post_dir}")
        
        pre_files = os.listdir(self.pre_dir)
        
        for pre_file in pre_files:
            if not self._is_image_file(pre_file):
                continue
                
            base_name = os.path.splitext(pre_file)[0]
            
            # Find corresponding post-event image
            post_file = self._find_matching_file(self.post_dir, base_name)
            if post_file is None:
                continue
            
            # Find corresponding mask (optional)
            mask_file = None
            if self.mask_dir and os.path.exists(self.mask_dir):
                mask_file = self._find_matching_file(self.mask_dir, base_name, 
                                                   extensions=['.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff'])
            
            sample = {
                'pre_path': os.path.join(self.pre_dir, pre_file),
                'post_path': os.path.join(self.post_dir, post_file),
                'mask_path': os.path.join(self.mask_dir, mask_file) if mask_file else None,
                'id': base_name
            }
            self.samples.append(sample)
        
        if len(self.samples) == 0:
            raise ValueError("No matching image pairs found!")
        
        print(f"Found {len(self.samples)} image pairs")
    
    def _is_image_file(self, filename):
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        return os.path.splitext(filename.lower())[1] in image_extensions
    
    def _find_matching_file(self, directory, base_name, extensions=None):
        """Find file with matching base name in directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        for ext in extensions:
            candidate = base_name + ext
            if os.path.exists(os.path.join(directory, candidate)):
                return candidate
        return None
    
    def _load_image(self, path):
        """Load and convert image to RGB"""
        try:
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Error loading image {path}: {e}")
    
    def _load_mask(self, path):
        """Load mask (supports .npy, .png, etc.)"""
        try:
            if path.endswith('.npy'):
                mask = np.load(path)
                if mask.ndim == 3:
                    mask = mask.squeeze()
            else:
                mask = np.array(Image.open(path).convert('L'))
            return mask.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Error loading mask {path}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        pre_image = self._load_image(sample['pre_path'])
        post_image = self._load_image(sample['post_path'])
        
        # Load mask if available
        if sample['mask_path']:
            mask = self._load_mask(sample['mask_path'])
        else:
            mask = np.zeros((pre_image.shape[0], pre_image.shape[1]), dtype=np.float32)
        
        # Apply transformations
        # Note: We apply same random augmentations to both images and mask
        if self.transform:
            # For consistency, we use the same random seed for both images
            augmented = self.transform(image=pre_image, mask=mask)
            pre_image = augmented['image']
            mask = augmented['mask']
            
            # Apply same transform to post image
            augmented_post = self.transform(image=post_image, mask=mask)
            post_image = augmented_post['image']
            # Use mask from first transform to ensure consistency
        
        # Convert mask to long tensor for loss computation
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return {
            'pre_image': pre_image,
            'post_image': post_image,
            'mask': mask,
            'id': sample['id']
        }


class InferenceDataset(Dataset):
    """
    Dataset for inference (no masks required)
    """
    
    def __init__(self, pre_dir, post_dir, transform=None, image_size=(224, 224)):
        self.dataset = BiTemporalDataset(
            pre_dir=pre_dir,
            post_dir=post_dir,
            mask_dir=None,
            transform=transform,
            image_size=image_size
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'pre_image': sample['pre_image'],
            'post_image': sample['post_image'],
            'id': sample['id']
        }


def get_transforms(mode='train', image_size=(224, 224), normalize=True):
    """
    Get data transformation pipeline
    
    Args:
        mode (str): 'train', 'val', or 'test'
        image_size (tuple): Target image size
        normalize (bool): Whether to apply ImageNet normalization
    
    Returns:
        albumentations.Compose: Transformation pipeline
    """
    if mode == 'train':
        transforms_list = [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], p=0.3),
        ]
    else:
        transforms_list = [
            A.Resize(image_size[0], image_size[1])
        ]
    
    if normalize:
        transforms_list.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)


def create_dataloaders(data_dir, batch_size=4, num_workers=4, image_size=(224, 224),
                      train_split=0.7, val_split=0.2, random_seed=42):
    """
    Create train/val/test dataloaders
    
    Args:
        data_dir (str): Root directory containing 'pre', 'post', and 'masks' subdirectories
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create dataset
    pre_dir = os.path.join(data_dir, 'pre')
    post_dir = os.path.join(data_dir, 'post')
    mask_dir = os.path.join(data_dir, 'masks')  # or 'labels', 'gt', etc.
    
    dataset = BiTemporalDataset(
        pre_dir=pre_dir,
        post_dir=post_dir,
        mask_dir=mask_dir,
        transform=get_transforms('train', image_size),
        image_size=image_size
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Update transforms for val/test
    val_transform = get_transforms('val', image_size)
    test_transform = get_transforms('test', image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    import tempfile
    
    # Create dummy data for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        pre_dir = os.path.join(temp_dir, 'pre')
        post_dir = os.path.join(temp_dir, 'post')
        os.makedirs(pre_dir)
        os.makedirs(post_dir)
        
        # Create dummy images
        dummy_image = Image.new('RGB', (256, 256), color='red')
        dummy_image.save(os.path.join(pre_dir, 'test.png'))
        dummy_image.save(os.path.join(post_dir, 'test.jpg'))
        
        # Test dataset
        dataset = BiTemporalDataset(pre_dir, post_dir)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Pre image shape: {sample['pre_image'].shape}")
            print(f"Post image shape: {sample['post_image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
