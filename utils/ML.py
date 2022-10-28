import torch
import time


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device, verbose=True):
    '''
        Train a model, given an optimizer and a number of epochs.
        Computes validation loss and accuracy after each epoch, and prints train/validation metrics.
        Returns per-iteration train and validation losses, for plotting.
    '''
    
    model.train()

    train_loss_log = []
    val_loss_log = []

    t = time.time()

    for i_epoch in range(num_epochs):

        epoch_loss = 0
        num_samples = 0

        for i_batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * len(y)
            num_samples += len(y)

        # calculate average loss of this epoch:
        mean_epoch_loss = epoch_loss/num_samples

        # calculate validation loss and accuracy:
        val_loss, val_acc = evaluate_model(model, val_dataloader, loss_fn, device=device)
        model.train()
        
        # print and save metrics:
        if verbose:
            print(f" epoch: {i_epoch+1:>2}, training loss: {mean_epoch_loss:.3f}, validation loss {val_loss:.3f}, validation accuracy {val_acc:.3f}")

        train_loss_log.append(mean_epoch_loss)
        val_loss_log.append(val_loss)

    # Calculate average time per epoch:
    time_per_epoch = (time.time()-t)/num_epochs
    
    if verbose:
        print(f'\nAverage time per epoch: {time_per_epoch:.3f}s')
    
    return train_loss_log, val_loss_log



def evaluate_model(model, dataloader, loss_fn, device):
    ''' Calculates average loss and accuracy over a dataset'''
    model.eval()
    
    num_correct = 0
    total_loss = 0
    num_samples = 0

    for i_batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Prediction on batch X:
        with torch.no_grad():
            pred = model(X)
            
        # Predicted class indexes:
        pred_idx = torch.argmax(pred, dim=1)
        
        # Batch loss:
        batch_loss = loss_fn(pred, y).item()

        total_loss += batch_loss * len(y)
        num_correct += torch.sum(pred_idx == y).item()
        num_samples += len(y)

    average_loss = total_loss/num_samples
    accuracy = num_correct/num_samples
        
    return average_loss, accuracy

def evaluate_ImageNet(model, dataloader, loss_fn, device):
    ''' Calculates average loss and accuracy and top-5 accuracy'''
    model.eval()
    
    num_correct = 0
    num_top5_correct = 0
    total_loss = 0
    num_samples = 0

    for i_batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Prediction on batch X:
        with torch.no_grad():
            pred = model(X)

        # Batch loss:
        batch_loss = loss_fn(pred, y).item()
            
        # top-5 predicted indexes:
        try:
            top5_pred_idx = torch.argsort(pred, descending=True)[:,:5]
        except:
            # if cannot be computed on mps
            pred = pred.to("cpu")
            y = y.to("cpu")
            top5_pred_idx = torch.argsort(pred, descending=True)[:,:5]

        # Predicted class indexes:
        pred_idx = top5_pred_idx[:, 0]
        # same as torch.argmax(pred, dim=1)

        total_loss += batch_loss * len(y)
        num_correct += torch.sum(pred_idx == y).item()
        num_top5_correct += torch.sum(top5_pred_idx == y[:, None])

        num_samples += len(y)

    average_loss = total_loss/num_samples
    accuracy = num_correct/num_samples
    top5_accuracy = num_top5_correct/num_samples
        
    return average_loss, accuracy, top5_accuracy


def train_inception(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device, verbose=True):
    '''
        Same as train_model() but adapted to Inception
        which returns 2 outputs (logits and auxilliary logits).
        Here we ignore auxilliary logits.
    '''
    
    model.train()

    train_loss_log = []
    val_loss_log = []

    t = time.time()

    for i_epoch in range(num_epochs):

        epoch_loss = 0
        num_samples = 0

        for i_batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred, _ = model(X)  # ignore auxiliary logits
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * len(y)
            num_samples += len(y)

        # calculate average loss of this epoch:
        mean_epoch_loss = epoch_loss/num_samples

        # calculate validation loss and accuracy:
        val_loss, val_acc = evaluate_inception(model, val_dataloader, loss_fn, device=device)
        model.train()
        
        # print and save metrics:
        if verbose:
            print(f" epoch: {i_epoch+1:>2}, training loss: {mean_epoch_loss:.3f}, validation loss {val_loss:.3f}, validation accuracy {val_acc:.3f}")

        train_loss_log.append(mean_epoch_loss)
        val_loss_log.append(val_loss)

    # Calculate average time per epoch:
    time_per_epoch = (time.time()-t)/num_epochs
    
    if verbose:
        print(f'\nAverage time per epoch: {time_per_epoch:.3f}s')
    
    return train_loss_log, val_loss_log



def evaluate_inception(model, dataloader, loss_fn, device):
    '''
        Same as evaluate_model() but adapted to Inception
        which returns 2 outputs (logits and auxilliary logits).
        Here we ignore auxilliary logits.
    '''
    model.eval()
    
    num_correct = 0
    total_loss = 0
    num_samples = 0

    for i_batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Prediction on batch X:
        with torch.no_grad():
            pred = model(X)
            
        # Predicted class indexes:
        pred_idx = torch.argmax(pred, dim=1)
        
        # Batch loss:
        batch_loss = loss_fn(pred, y).item()

        total_loss += batch_loss * len(y)
        num_correct += torch.sum(pred_idx == y).item()
        num_samples += len(y)

    average_loss = total_loss/num_samples
    accuracy = num_correct/num_samples
        
    return average_loss, accuracy




def print_overall_metrics(model, dataloaders, loss_fn, device):
    ''' Prints loss and accuracy for train, validation and test sets'''
    average_loss, accuracy = evaluate_model(model, dataloaders[0], loss_fn, device=device)
    print(f"train loss:      {average_loss:.5f}, accuracy: {accuracy:.5f}")

    average_loss, accuracy = evaluate_model(model, dataloaders[1], loss_fn, device=device)
    print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy:.5f}")

    average_loss, accuracy = evaluate_model(model, dataloaders[2], loss_fn, device=device)
    print(f"test loss:       {average_loss:.5f}, accuracy: {accuracy:.5f}")