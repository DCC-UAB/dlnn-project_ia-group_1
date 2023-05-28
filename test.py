import wandb
import torch

def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, text, labels in test_loader:
            images, text, labels = images.to(device), text.to(device), labels.to(device)
            outputs = model(images, text)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        torch.save(model.state_dict(),'/home/xnmaster/dlnn-project_ia-group_1/CheckPoints/model_state_dict_ADAMW_SGD_DROPUT_2.pth')
        wandb.save('/home/xnmaster/dlnn-project_ia-group_1/CheckPoints/model_state_dict_ADAMW_SGD_DROPUT_2.pth')