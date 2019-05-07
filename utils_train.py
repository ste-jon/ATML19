import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_loader, optimizer, loss_fn, print_every=100):
    '''
    Trains the model for one epoch
    '''
    model.train()
    losses = []
    n_correct = 0
    n_correct_top3 = 0
    for iteration, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
#         if iteration % print_every == 0:
#             print('Training iteration {}: loss {:.4f}'.format(iteration, loss.item()))
        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()
        sorted, indices = output.sort(1 ,descending=True)
        n_correct_top3 += torch.sum(labels == indices[:,1]).item()
        n_correct_top3 += torch.sum(labels == indices[:,2]).item()

    accuracy = 100.0 * n_correct / len(train_loader.dataset)
    n_correct_top3 += n_correct;
    accuracy_top3 = 100.0 * n_correct_top3 / len(train_loader.dataset)
    return np.mean(np.array(losses)), accuracy, accuracy_top3
            
def test(model, test_loader, loss_fn):
    '''
    Tests the model on data from test_loader
    '''
    model.eval()
    test_loss = 0
    n_correct = 0
    n_correct_top3 = 0
    n_per_label = np.zeros(32)
    n_correct_per_label = np.zeros(32)
    n_correct_per_label_top3 = np.zeros(32)
    cnt = 0;
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()
            sorted, indices = output.sort(1 ,descending=True)
            n_correct_top3 += torch.sum(labels == indices[:,1]).item()
            n_correct_top3 += torch.sum(labels == indices[:,2]).item()
            for it, label in enumerate(labels):
                n_per_label[label.item()] += 1
                n_correct_per_label[label.item()] += (label == indices[it,0]).item()
                n_correct_per_label_top3[label] += (label == indices[it,0]).item()
                n_correct_per_label_top3[label] += (label == indices[it,1]).item()
                n_correct_per_label_top3[label] += (label == indices[it,2]).item()

    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    n_correct_top3 += n_correct
    accuracy_top3 = 100.0 * n_correct_top3 / len(test_loader.dataset)
  
    accuracy_per_label = 100 * np.divide(n_correct_per_label, n_per_label)
    accuracy_per_label_top3 = 100 * np.divide(n_correct_per_label_top3, n_per_label)
    
    return average_loss, accuracy, accuracy_top3, accuracy_per_label, accuracy_per_label_top3


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, scheduler=None):
    train_losses, train_accuracies, train_accuracies_top3 = [], [], []
    val_losses, val_accuracies, val_accuracies_top3 = [], [], []

    for epoch in range(n_epochs):
        train_loss, train_accuracy, train_accuracy_top3 = train(model, train_dataloader, optimizer, loss_fn)
        val_loss, val_accuracy, val_accuracy_top3, val_accuracy_per_label, val_accuracy_per_label_top3 = test(model, val_dataloader, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_accuracies_top3.append(train_accuracy_top3)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_accuracies_top3.append(val_accuracy_top3)
        if scheduler:
            scheduler.step() # argument only needed for ReduceLROnPlateau
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, top3: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}, top3: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                          train_losses[-1],
                                                                                                          train_accuracies[-1],
                                                                                                          train_accuracies_top3[-1],
                                                                                                          val_losses[-1],
                                                                                                          val_accuracies[-1],
                                                                                                          val_accuracies_top3[-1]))
    
    return train_losses, train_accuracies, val_losses, val_accuracies, val_accuracy_per_label, val_accuracy_per_label_top3