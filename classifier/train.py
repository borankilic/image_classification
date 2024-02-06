"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
from torch import optim
from torchvision import transforms
import time
import data_setup
import engine
import model_builder
import utils

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01

# Setup directories
train_dir = "../data/split_dataset/train"
test_dir = "../data/split_dataset/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"



# Create transforms
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(size=(227, 227), antialias=False)])


if __name__ == '__main__':
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names, test_dataset = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.VAVLAB_NET_V1(
        input_shape=3,
        output_shape=4
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.7)


    # Start training with help from engine.py
    results = engine.train_test_loop(model=model,
                                     train_dataloader=train_dataloader,
                                     test_dataloader=test_dataloader,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     epochs=NUM_EPOCHS,
                                     device=device,
                                     scheduler=scheduler)

    # Plot loss curves
    utils.plot_loss_curves(results=results)

    # Visualize model guesses
    utils.visualise_model_guesses(model=model, test_dataset=test_dataset, device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="../models",
                     model_name="VAVLAB_NET_V1.pth")
