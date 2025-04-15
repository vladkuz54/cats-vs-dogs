import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from helper_functions import train, plot_loss_curves, pred_and_plot_image
from model import TinyVGG

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path('data/')
    image_path = data_path / 'cat-dog'

    train_dir = image_path / "train"
    test_dir = image_path / 'test'

    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=15),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                                transform=train_transform)

    test_data = datasets.ImageFolder(root=test_dir,
                                            transform=test_transform)

    # BATCH_SIZE = 32
    # NUM_WORKERS = os.cpu_count()
    #
    # train_dataloader = DataLoader(dataset=train_data,
    #                                         batch_size=BATCH_SIZE,
    #                                         shuffle=True,
    #                                         num_workers=NUM_WORKERS)
    #
    # test_dataloader = DataLoader(dataset=test_data,
    #                                     batch_size=BATCH_SIZE,
    #                                     shuffle=False,
    #                                     num_workers=NUM_WORKERS)
    #
    # model = TinyVGG(input_shape=3,
    #                 hidden_units=10,
    #                 output_shape=len(train_data.classes)).to(device)
    #
    # NUM_EPOCHS = 10
    #
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=0.01)
    #
    # model_results = train(model=model,
    #                       train_dataloader=train_dataloader,
    #                       test_dataloader=test_dataloader,
    #                       optimizer=optimizer,
    #                       loss_fn=loss_fn,
    #                       epochs=NUM_EPOCHS)
    #
    # plot_loss_curves(model_results)

    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    MODEL_NAME = 'cats-vs-dogs.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    #
    # torch.save(obj=model.state_dict(),
    #            f=MODEL_SAVE_PATH)

    loaded_model = TinyVGG(input_shape=3,
                            hidden_units=10,
                            output_shape=len(train_data.classes))

    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    loaded_model.to(device)

    custom_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64))
    ])

    # custom_image_path =

    # pred_and_plot_image(model=loaded_model,
    #                     image_path=custom_image_path,
    #                     class_names=train_data.classes,
    #                     transform=custom_image_transform,
    #                     device=device)