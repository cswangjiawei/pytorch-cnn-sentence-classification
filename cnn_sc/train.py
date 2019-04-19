import torch.nn.functional as F
import torch


def train_model(train_iter, model, criterion, optimizer):
    model.train()
    total_loss = 0.

    for batch in train_iter:
        model.zero_grad()
        word_input = batch.text
        target = batch.label - 2
        tag_socre = model(word_input)
        tag_socre = tag_socre.view(-1, tag_socre.size(1))
        loss = criterion(tag_socre, target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('total_loss:', total_loss)
    return total_loss


def evaluate(val_or_test_iter, model):
    model.eval()
    correct_num = 0
    total_num = 0

    for batch in val_or_test_iter:
        word_input = batch.text
        target = batch.label - 2
        target = target.view(-1)
        total_num += len(target)
        tag_score = model(word_input)
        tag_score = F.softmax(tag_score, dim=1)
        _, preds = torch.max(tag_score, 1)
        correct_num += torch.sum((preds == target)).item()

    acc = (correct_num/total_num) * 100
    return acc