import torch
from pathlib import Path
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):

    """ CustomDataset takes 2 positional arguments root_dir: [str] and transform: [torchvision.transforms].
    It returns an image tensor and the corresponding label[int] on the given index by overwriting the __getitem__() method"""

    def __init__(self, root_dir, transform=None):
        super().__init__()
        ##You can add additional parameters if you want
        self.root_dir = root_dir
        self.image_paths = list(Path(root_dir).glob("*/*.jpg"))
        self.transform = transform


    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """ Finds the class folder names in a target directory."""
        # 1. Get the class names byu scanning the target directory
        classes = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())

        # 2. Raise an error if names could not be found
        if not classes:
          raise FileNotFoundError(f"Couldn't find any classes in {self.root_dir}...please check file structure")

        # 3. Create a dictionary of index labels (computers prefer numbers rather than strings as labels)
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return class_to_idx, classes




    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        ##This is the part where you obtain data samples from dataset. It is essential for dataloader to operate.
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        ##resize image to 224,224 by using Image.resize
        image = image.resize((224,224), resample=Image.NEAREST )

        ## Define a mapping that maps classes to a number, i.e Niagara--->0 Eiffel---->1 Taj Mahal--->3 Brooklyn Bridge--->4 Order can be changed it is up to you.
        ##IMPORTANT NOTE :: YOUR IMAGE AND LABEL MUST BE PYTORCH TENSOR!!!!!
        class_to_idx, classes = self.find_classes()
        class_name = self.image_paths[idx].parent.name # expects path in format: data_folder/class_name/image.jpg
        label = torch.tensor(class_to_idx[class_name])

        if self.transform:
            image = self.transform(image)
        else:
          image = torch.from_numpy(image)
          image = torch.permute(image, (2,1,0))
#        image = image.type(torch.FloatTensor)
        return image, label