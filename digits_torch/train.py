from torch.nn import functional as F


def train_loop(model, optimizer, train_dataloader, epochs=5, log_inter=100):
    data_len = len(train_dataloader.dataset)
    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            # zero the optimizer accumulator
            optimizer.zero_grad()
            # compute output
            output = model(data.cuda())
            # compute loss
            loss = F.cross_entropy(output, target.cuda())
            # compute the gradient of loss wrt parameters using autodiff
            loss.backward()
            # update weights
            optimizer.step()

            if batch_idx % log_inter == 0:
                print(f"Epoch : {epoch}, [{batch_idx * len(data)}/{data_len}], loss: {loss.item()} ")

