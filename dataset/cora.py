import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class CoraDataset(InMemoryDataset):
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    geom_gcn_url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/" "geom-gcn/master"
    )

    def __init__(
        self,
        root: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        noise_rate: float = 0.0,
        sigma: float = 0.0,
    ):
        self.name = "Cora"
        self.noise_rate = noise_rate
        self.sigma = sigma

        self.split = split.lower()
        assert self.split in ["public", "full", "geom-gcn", "random"]
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

        if split == "full":
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == "random":
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val : num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "raw")
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "processed")
        return osp.join(
            self.root,
            f"{self.name}-noise-{self.noise_rate}-simma-{self.sigma}",
            "processed",
        )

    @property
    def raw_file_names(self) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name.lower()}.{name}" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for name in self.raw_file_names:
            download_url(f"{self.url}/{name}", self.raw_dir)
        if self.split == "geom-gcn":
            for i in range(10):
                url = f"{self.geom_gcn_url}/splits/{self.name.lower()}"
                download_url(f"{url}_split_0.6_0.2_{i}.npz", self.raw_dir)
        # pass

    def process(self):
        print("process data")
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == "geom-gcn":
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f"{self.name.lower()}_split_0.6_0.2_{i}.npz"
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits["train_mask"]))
                val_masks.append(torch.from_numpy(splits["val_mask"]))
                test_masks.append(torch.from_numpy(splits["test_mask"]))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)

        # add noise
        feat = data.x
        noise = torch.randn_like(feat) * self.sigma
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        train_noise_mask = sample_mask(train_mask, self.noise_rate).int()
        val_noise_mask = sample_mask(val_mask, self.noise_rate).int()
        test_noise_mask = sample_mask(test_mask, self.noise_rate).int()
        noise_mask = train_noise_mask + val_noise_mask + test_noise_mask
        noise_mask = noise_mask.bool()

        num_words = (feat > 0).sum(dim=1)
        feat = num_words.unsqueeze(1).repeat(1, feat.size(-1)) * feat

        data.x = feat + noise * noise_mask.unsqueeze(1).repeat(1, feat.size(-1))

        # print(feat)
        # print(noise * noise_mask.unsqueeze(1).repeat(1, feat.size(-1)))
        # print(data.x)

        print("Save Data ...")
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"


def sample_mask(mask: torch.Tensor, percentage: float):
    # print(mask)
    one_indices = torch.nonzero(mask == 1).squeeze()
    # print(one_indices)
    num_samples = int(percentage * len(one_indices))
    selected_indices = torch.randperm(len(one_indices))[:num_samples]
    new_mask = torch.zeros_like(mask)
    new_mask[one_indices[selected_indices]] = 1
    return new_mask


if __name__ == "__main__":
    dataset = CoraDataset(root="data/Cora", noise_rate=1, sigma=1.1)
