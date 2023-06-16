from eval_script import compute_revenue
import torch
import numpy as np
from BirdCrossEtropyLoss import BirdCrossEntropyLoss

def smooth_sample_zero(pred):
    print(pred)
    max_lbl = np.bincount(pred)
    max_lbl[0] = 0
    bird = np.argmax(max_lbl)

    pred[pred != 0] = bird
    print(pred)
    print("-----------------------------------")
    return pred

def train(dataloader, model, loss_fn, optimizer, progress_steps):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y, i) in enumerate(dataloader):
        # Compute prediction and loss

        # TODO: Scale preds that are not 0 or cowpig by 1.2 for example
        pred = model(X)
        loss = loss_fn(pred, y, i)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if progress_steps != None and batch % progress_steps == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, show_progress):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    money = 0
    smooth_money = 0

    with torch.no_grad():
        for X, y, i in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y, i).item()
            smooth_pred = smooth_sample_zero(torch.argmax(pred, dim=1).cpu().numpy())

            money += compute_revenue(torch.argmax(pred, dim=1).cpu(), torch.argmax(y, dim=1).cpu())
            smooth_money += compute_revenue(torch.from_numpy(smooth_pred), torch.argmax(y, dim=1).cpu())
    test_loss /= num_batches

    if show_progress:
        print(f"Avg loss: {test_loss:>8f}, Money saved: {money:.2f}, Smoothed: {smooth_money:.2f}$\n")
    return test_loss, smooth_money

def eval(model, train_dataloader, test_dataloader, device, max_epochs, name, lr=0.01):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    loss_fn = BirdCrossEntropyLoss(device)
    lowest_loss = np.infty
    highest_money = -np.infty
    stop_criterion = 0
    for t in range(max_epochs):
        if t % 10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, None)
        loss, money = test(test_dataloader, model, loss_fn, True)
        scheduler.step(loss)
        if money > highest_money:
            highest_money = money
            torch.save(model.state_dict(), f'Models/{name}')
            stop_criterion = 0
        else:
            stop_criterion += 1
        # if loss > lowest_loss:
        #     stop_criterion += 1
        # else:
        #     torch.save(model.state_dict(), f'Models/{name}')
        #     lowest_loss = loss
        #     stop_criterion = 0
        if stop_criterion >= 30:
            break
    model.load_state_dict(torch.load(f'Models/{name}'))
    model.eval()
    print("Done:")
    test(test_dataloader, model, loss_fn, True)