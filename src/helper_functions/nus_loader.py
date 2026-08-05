from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

defaultTransform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

class NUSWIDELoader:
    def __init__(self, root, img_root, split, included=[], transform=defaultTransform):
        self.root = root
        self.img_root = img_root
        self.included_cats = included
        self.transform = transform
        self.img_ids = np.load(os.path.join(root, 'formatted_{}_images.npy'.format(split)))
        self.label_matrix = np.load(os.path.join(root, 'formatted_{}_labels.npy'.format(split)))

        indices = np.array([], dtype=np.int64)
        if self.included_cats == []:
            indices = np.append(indices, np.arange(len(self.img_ids)))
        else:
            for cid in self.included_cats:
                indices = np.append(indices, np.where(self.label_matrix[:, cid] == 1)[0])
            indices = np.unique(indices)
        self.selected_img_ids = self.img_ids[indices]
        self.selected_label = self.label_matrix[indices]

    def __len__(self):
        return len(self.selected_img_ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_root, self.selected_img_ids[idx])
        with Image.open(image_path) as img_raw:
            img = img_raw.convert('RGB')
        label = torch.FloatTensor(np.copy(self.selected_label[idx, :]))
        return self.transform(img), label


if __name__ == '__main__':
    DATASETS_ROOT = './datasets/nuswide'
    IMAGE_ROOT =  './datasets/nuswide/Flickr'
    dataset = NUSWIDELoader(root=DATASETS_ROOT, img_root=IMAGE_ROOT, split='train', included=[0,1,2,3])
    print(dataset[0])