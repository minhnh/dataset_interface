from PIL import Image
import torchvision.transforms as transforms

def get_transforms():
    transform = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()])
    return transform

def get_grayscale_image_tensor(img_path):
    img = Image.open(img_path)
    img = img.convert("L")
    img = get_transforms()(img)
    img.unsqueeze_(0)
    return img
