import torch
from .metrics import scores


def train_one_epoch(model, loader, criterion, optimizer, device,
                    return_predictions=False):
    model.train()
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * yb.size(0)
        pred = logits.detach().argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true += yb.detach().cpu().tolist()
        y_pred += pred.cpu().tolist()
    if not total:
        raise ValueError('Empty training loader')
    result = (running / total, correct / total)
    return (*result, y_true, y_pred) if return_predictions else result


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true += yb.cpu().tolist()
        y_pred += pred.cpu().tolist()
    if not total:
        raise ValueError('Empty evaluation loader')
    return running / total, correct / total, y_true, y_pred


def run_epoch(model, loader, device, num_classes, optimizer=None):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is not None:
        loss, _, truth, predictions = train_one_epoch(
            model,loader,criterion,optimizer,device,return_predictions=True)
    else:
        loss, _, truth, predictions = evaluate(model,loader,criterion,device)
    result = scores(truth,predictions,num_classes)
    result['loss'] = loss
    return result, truth, predictions
