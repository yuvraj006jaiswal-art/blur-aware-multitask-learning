import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

# -------------------------------------------------------------------------
# Think of BlurDataset like a "Photo Filter Box".
# It takes an existing dataset (like CIFAR-10) and applies blur on demand.
# -------------------------------------------------------------------------
class BlurDataset(Dataset):
    def __init__(self, dataset, blur_level=0):
        """
        1. __init__ is just the SETUP step.
        It runs ONCE when you create the dataset.
        """
        self.dataset = dataset
        self.blur_level = blur_level

        # Define the blur tools (scaled appropriately for 224x224 images):
        # level 0 = clean (no blur)
        # level 1 = mild blur
        # level 2 = heavy blur
        if blur_level == 1:
            self.blur_filter = transforms.GaussianBlur(kernel_size=15, sigma=3.0)
        elif blur_level == 2:
            self.blur_filter = transforms.GaussianBlur(kernel_size=35, sigma=8.0)
        else:
            self.blur_filter = None

    def __len__(self):
        """
        2. __len__ tells PyTorch: "How many images are in this dataset?"
        Whenever someone calls len(my_dataset), Python runs this method.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        3. __getitem__ tells PyTorch: "What should I return for image #index?"
        Whenever someone asks for my_dataset[index], Python runs this method.
        """
        image, class_label = self.dataset[index]

        # If blur is needed, apply it to the image
        if self.blur_filter is not None:
            image = self.blur_filter(image)

        # Return: (the image, the original class label, and the blur level: 0, 1, or 2)
        return image, class_label, self.blur_level


def create_blur_datasets(train_data, test_data):
    """
    Helper function: Creates all the train and test splits in one simple call!
    """
    # 1. Test sets for each blur level (to evaluate how well the model handles each level)
    test_clean = BlurDataset(test_data, blur_level=0)
    test_mild  = BlurDataset(test_data, blur_level=1)
    test_heavy = BlurDataset(test_data, blur_level=2)

    # 2. Train set: Mix clean + mild + heavy together so the model learns from all of them
    train_clean = BlurDataset(train_data, blur_level=0)
    train_mild  = BlurDataset(train_data, blur_level=1)
    train_heavy = BlurDataset(train_data, blur_level=2)
    train_mixed = ConcatDataset([train_clean, train_mild, train_heavy])

    return {
        "train_mixed": train_mixed,
        "test_clean": test_clean,
        "test_mild": test_mild,
        "test_heavy": test_heavy
    }

