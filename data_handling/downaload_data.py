import torchvision
import torchvision.transforms as transforms

def get_data():
    
    base_transform = transforms.Compose([
    transforms.Resize((224, 224)),      
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=base_transform)

    test_data  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=base_transform)

    return train_data, test_data