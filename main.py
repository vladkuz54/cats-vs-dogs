import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from helper_functions import train, plot_loss_curves
from model import TinyVGG

if __name__ == "__main__":  # Ensure safe multiprocessing on Windows
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path('data/')
    image_path = data_path / 'cat-dog'

    train_dir = image_path / "train"
    test_dir = image_path / 'test'

    train_transform_trivial = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=15),
        transforms.ToTensor()
    ])

    test_transform_simple = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_data_augmented = datasets.ImageFolder(root=train_dir,
                                                transform=train_transform_trivial)

    test_data_simple = datasets.ImageFolder(root=test_dir,
                                            transform=test_transform_simple)

    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    model = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(train_data_augmented.classes)).to(device)

    NUM_EPOCHS = 20

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)

    model_results = train(model=model,
                          train_dataloader=train_dataloader_augmented,
                          test_dataloader=test_dataloader_simple,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=NUM_EPOCHS)

    plot_loss_curves(model_results)

    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    MODEL_NAME = 'cats-vs-dogs.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)
