import torch
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import random


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def plot_loss_curves(results: Dict[str, List[torch.tensor]]):
    """ Plots training curves of a results dictionary """
    # Get the loss values of results dictionary
    train_loss = np.stack([loss.cpu().detach().numpy() for loss in results['train_loss']])
    test_loss = results['test_loss']

    # Get the acc values of resutls dict
    train_accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def visualise_model_guesses(model, test_dataset, device):
    img_indexes = random.sample(range(len(test_dataset)), 9)
    fig = plt.figure(figsize=(12, 12))
    for i, idx in enumerate(img_indexes):
        raw_img, target_label = test_dataset[idx]
        target_img = raw_img.to(device).type(torch.float32)
        target_img = torch.unsqueeze(target_img, dim=0)
        pred_label = torch.argmax(torch.softmax(model(target_img), dim=1), dim=1)
        class_to_idx, classes = test_dataset.find_classes()
        target_class = classes[target_label]
        pred_class = classes[pred_label]
        raw_img = torch.permute(raw_img, (2, 1, 0))
        fig.add_subplot(3, 3, i + 1)
        plt.imshow(raw_img)
        title_text = f"Pred: {pred_class} \n Truth: {target_class}"

        if pred_label == target_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")
        plt.axis("off")
