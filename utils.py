from torchvision import transforms


path = "D:\\Datasets\\Computer Vision\\vegetable\\train\\Potato"
img_size = 56
batch_size = 32
in_channels = 3
noise_dim = 200
lr = 0.001
epochs = 10
dis_epochs = 1
alpha = 5
epsilon = 0.1

transform = transforms.Compose([
    transforms.ToTensor()
])
