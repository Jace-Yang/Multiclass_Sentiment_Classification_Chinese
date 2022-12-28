import torch
import time
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm.notebook import tqdm
import pandas as pd

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, pretrain_path, hidden_size):
        '''
        pretrain_path: local or hugging-face path, e.g '/roberta-wwm-ext pretrain/'
        '''
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = True  # Allow all parameters to be updated
        self.fc = nn.Linear(hidden_size, num_classes)   # A layer to calculate logits of 6 ouput classes from 768 (hidden size of BERT)
            # Note: We are going to use Cross-EntropyLoss with a softmax “embedded”.
    def forward(self, x, token_type_ids, attention_mask):
        context = x  # Input sentence
        segments = token_type_ids
        mask = attention_mask  # Only mask the padding part
        _, pooled = self.bert(context, token_type_ids=segments, attention_mask=mask)
        logits = self.fc(pooled) # probability of 6 classes
        return logits

class StackedClassifier(nn.Module):
    def __init__(self, models, stack_weights, device):
        super(StackedClassifier, self).__init__()
        if 'Stacked Model' in models.keys():
            del models['Stacked Model']
        self.models = models
        self.stack_weights = torch.tensor([[stack_weights[model_name] for model_name, _ in self.models.items()]]).squeeze().to(device)
    def forward(self, x, token_type_ids, attention_mask):
        predicts = [torch.softmax(model(x, token_type_ids, attention_mask), dim=1).unsqueeze(2) for _, model in self.models.items()]
        x = (torch.cat(predicts, axis=2) * self.stack_weights).sum(axis=2)
        return x

def train(model, model_name, train_loader, test_loader, optimizer, device, n_epoch, model_saving_path):
    '''Train the model
    '''
    model.train()
    best_f1 = 0.0
    training_loss = []
    training_acc = []
    training_f1 = []
    validation_loss = []
    validation_acc = []
    validation_f1 = []
    time_usage = []
    epochs = list(range(1, n_epoch + 1))
    for epoch in tqdm(epochs):
        batch_idx = 0
        running_loss = 0
        training_start_time = time.time()
        pred = []
        y_train = []
        for (word_ids, token_types, attention_masks, y) in tqdm(train_loader):
            word_ids, token_types, attention_masks, y = word_ids.to(device), token_types.to(device), attention_masks.to(device), y.to(device)
            y_pred = model(word_ids, token_type_ids=token_types, attention_mask=attention_masks)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y.squeeze()) # Calculate Loss
            loss.backward()
            optimizer.step()
            # Logging the loss and accuracy
            running_loss += loss.item()
            pred += y_pred.argmax(dim=1).tolist() # Get the maximum probability
            y_train += y.squeeze().tolist()
            batch_idx += 1
            # Print Every 250 batch
            if(batch_idx + 1) % 250 == 0:
                print('Epoch: {} [{}/{} ({:.2f}%)]\tBatch Loss: {:.6f}\tAvg Loss: {:.6f}\t'.format(
                    epoch, 
                    batch_idx * len(word_ids),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    running_loss / batch_idx))
        # Compute time cost
        time_cost = time.time() - training_start_time
        time_usage.append(time_cost)
        print(f'Epoch {epoch} finished, took {time_cost:.1f}s')

        # Logging loss and accuracy, average on every updates(batches) in the training stage
        training_loss.append(running_loss / len(train_loader))
        training_acc.append(accuracy_score(y_train, pred))
        training_f1.append(f1_score(y_train, pred, average='macro'))
        
        # Evaluate performance on testset
        val_loss, val_acc, val_f1, _ = test(model, test_loader, device)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        validation_f1.append(val_f1)

        # Keep Best model base on f1
        if best_f1 < val_f1:
            torch.save(model.state_dict(), model_saving_path)
            best_f1 = val_f1

    # Output logs after all epoches
    progress_log = pd.DataFrame({'Model': model_name,
                                 'Epoch': epochs,
                                 'training_loss': training_loss,
                                 'training_acc': training_acc,
                                 'training_f1': training_f1,
                                 'validation_loss': validation_loss,
                                 'validation_acc': validation_acc,
                                 'validation_f1': validation_f1,
                                 'time_usage': time_usage
                                 })
    return progress_log


def test(model, test_loader, device):
    '''Evaluate the model
    '''
    model.eval()
    test_loss = 0.0
    y_test = []
    pred = []
    inference_start = time.time()
    for (word_ids, token_types, attention_masks, y) in test_loader:
        word_ids, token_types, attention_masks, y = word_ids.to(device), token_types.to(device), attention_masks.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(word_ids, token_type_ids=token_types, attention_mask=attention_masks)
        test_loss += F.cross_entropy(y_, y.squeeze()).item()
        y_test += y.squeeze().tolist()
        pred += y_.argmax(dim=1).tolist() # Obtain the maximum probability
    inference_time = time.time() - inference_start
    test_loss /= len(test_loader)
    test_correct = accuracy_score(y_test, pred, normalize=False)
    test_acc = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred, average='macro')
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Macro F1: {:.4f}%, took {:.1f}s'.format(
          test_loss, test_correct, len(test_loader.dataset),
          100. * test_acc,
          100. * test_f1,
          inference_time))
    return test_loss, test_acc, test_f1, inference_time

if __name__ == "__main__":
    pass