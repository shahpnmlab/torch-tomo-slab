# src/torch_tomo_slab/data/sampling.py

import torch
import torchio as tio
from torch.utils.data import Dataset
from typing import Dict, List, Iterator, Union

class TorchioPatchSampler:
    """
    A wrapper to apply a TorchIO patch sampler to a dictionary of tensors.
    This class is now designed to handle two modes:
    1. Using a pre-instantiated sampler (like LabelSampler).
    2. Using a string identifier ('grid') to instantiate GridSampler per-subject.
    """
    def __init__(self, sampler: Union[tio.data.sampler.PatchSampler, str], patch_size: int, overlap: int):
        """
        Initializes the patch sampler wrapper.

        Args:
            sampler: Either an instance of a TorchIO sampler (e.g., tio.LabelSampler)
                     or the string 'grid' to indicate GridSampling.
            patch_size: The size of the patches to extract.
            overlap: The overlap between adjacent patches.
        """
        self.sampler = sampler
        self.patch_size = patch_size
        self.overlap = overlap
        pad_amount = overlap // 2
        self.padder = tio.Pad((pad_amount, pad_amount, pad_amount, pad_amount, 0, 0))

    def __call__(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Takes a 2D sample, converts it to a 3D TorchIO subject, and extracts patches.
        """
        image_2d, label_2d = sample['image'], sample['label']
        image_3d, label_3d = image_2d.unsqueeze(1), label_2d.unsqueeze(1)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_3d),
            label=tio.LabelMap(tensor=label_3d),
        )
        padded_subject = self.padder(subject)
        
        # --- LOGIC TO HANDLE DIFFERENT SAMPLER TYPES ---
        if isinstance(self.sampler, str) and self.sampler == 'grid':
            # Instantiate GridSampler here, where the subject is available
            grid_sampler = tio.GridSampler(
                subject=padded_subject,
                patch_size=self.patch_size,
                patch_overlap=self.overlap,
            )
            patch_generator = grid_sampler
        else:
            # For pre-instantiated samplers like LabelSampler
            patch_generator = self.sampler(padded_subject)
        
        patches = []
        for patch in patch_generator:
            image_patch = patch['image'][tio.DATA].squeeze(1)
            label_patch = patch['label'][tio.DATA].squeeze(1)
            patches.append({'image': image_patch, 'label': label_patch})
            
        return patches

class IterablePatchDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset that yields patches from a subjects dataset.
    Handles multi-worker data distribution.
    """
    def __init__(self, subjects_dataset: Dataset, patch_sampler: TorchioPatchSampler):
        super().__init__()
        self.subjects_dataset = subjects_dataset
        self.patch_sampler = patch_sampler

    def __len__(self) -> int:
        return len(self.subjects_dataset)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        
        all_subject_indices = list(range(len(self.subjects_dataset)))
        subject_indices_for_this_worker = all_subject_indices[worker_id::num_workers]

        patches_from_all_subjects = []
        for subject_idx in subject_indices_for_this_worker:
            subject_sample = self.subjects_dataset[subject_idx]
            subject_patches = self.patch_sampler(subject_sample)
            patches_from_all_subjects.extend(subject_patches)
        
        generator = torch.Generator()
        if worker_info:
            generator.manual_seed(worker_info.seed)
        
        perm = torch.randperm(len(patches_from_all_subjects), generator=generator)
        for idx in perm:
            yield patches_from_all_subjects[idx]
