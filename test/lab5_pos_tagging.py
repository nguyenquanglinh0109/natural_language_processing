import pyconll
from src.models.pos_tagging import SimpleRNNForTokenClassification
from src.loader.pos_dataloader import POSDataset, custom_collate_fn
from src.core.build_vocab import Vocabulary, get_data
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np

def get_dataloader(data_path: str, word_to_idx, tag_to_idx):
    sentences = get_data(data_path)
    dataset = POSDataset(sentences, word_to_idx, tag_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    return dataloader

def compute_loss_and_acc(pred, tag, criterion):
    # pred: (batch, seq_len, num_tags)
    # tag:  (batch, seq_len)

    batch, seq_len, num_tags = pred.shape
    
    pred_flat = pred.reshape(-1, num_tags)      # (B*L, num_tags)
    tag_flat = tag.reshape(-1)                  # (B*L)

    loss = criterion(pred_flat, tag_flat)

    # accuracy (bỏ padding)
    mask = tag_flat != -1
    pred_label = torch.argmax(pred_flat, dim=1)

    correct = (pred_label[mask] == tag_flat[mask]).sum().item()
    total = mask.sum().item()

    acc = correct / total if total > 0 else 0.0

    return loss, acc

def train_epoch(model, optimizer, criterion, dataloader, epoch):
    model.train()
    loss_epoch, acc_epoch = [], []

    for sentence, tag in tqdm(dataloader, desc="Training..."):
        optimizer.zero_grad()

        pred = model(sentence)   # (batch, seq_len, num_tags)

        loss, acc = compute_loss_and_acc(pred, tag, criterion)

        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
        acc_epoch.append(acc)

    return np.mean(loss_epoch), np.mean(acc_epoch)

def evaluate_epoch(model, criterion, dataloader, epoch):
    model.eval()
    loss_epoch, acc_epoch = [], []

    with torch.no_grad():
        for sentence, tag in tqdm(dataloader, desc="Evaluating..."):
            pred = model(sentence)

            loss, acc = compute_loss_and_acc(pred, tag, criterion)

            loss_epoch.append(loss.item())
            acc_epoch.append(acc)

    return np.mean(loss_epoch), np.mean(acc_epoch)

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs):
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in tqdm(range(num_epochs), desc="Training...."):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, epoch)
        dev_loss, dev_acc = evaluate_epoch(model, criterion, val_loader, epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(dev_loss)
        val_accs.append(dev_acc)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")
    
def predict_sentence(sentence, vocab, model_path, embedding_dim = 100, hidden_size = 256):
    try:
        print("Loading model...")
        model = SimpleRNNForTokenClassification(vocab_size=len(vocab.word_to_idx), embedding_dim=embedding_dim, hidden_size=hidden_size, output_dim=len(vocab.tag_to_idx))
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except Exception as e:
        print("Error load model: ", e)
        return
    
    tokens = sentence.split()
    token_ids = [vocab.word_to_idx.get(token, vocab.word_to_idx["<UNK>"]) for token in tokens]
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    pred = model(token_ids)
    pred = torch.argmax(pred, dim=2)
    pred = pred.squeeze().tolist()
    idx_to_tag = {v: k for k, v in vocab.tag_to_idx.items()}
    
    return [(tokens[i], idx_to_tag[pred[i]]) for i in range(len(tokens))]
    
    
def main():
    data_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.conllu"
    vocab = Vocabulary(data_path)
    vocab.build_index()
    word_to_idx = vocab.word_to_idx
    tag_to_idx = vocab.tag_to_idx
    
    train_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.conllu"
    test_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-test.conllu"
    dev_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-dev.conllu"
    
    train_loader = get_dataloader(train_path, word_to_idx, tag_to_idx)
    test_loader = get_dataloader(test_path, word_to_idx, tag_to_idx)
    val_loader = get_dataloader(dev_path, word_to_idx, tag_to_idx)
    
    vocab_size = len(word_to_idx)
    embedding_dim = 100
    hidden_size = 256
    output_dim = len(tag_to_idx)
    model = SimpleRNNForTokenClassification(vocab_size, embedding_dim, hidden_size, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    num_epochs = 10
    train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs)
    
    test_loss, test_acc = evaluate_epoch(model, criterion, test_loader, num_epochs)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    save_path = r"trained_model\pos_tagging_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
def sample_test():
    data_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.conllu"
    vocab = Vocabulary(data_path)
    vocab.build_index()
    model_path = r"trained_model\pos_tagging_model.pth"
    
    inp_text = input("Sample test: ")
    result = predict_sentence(inp_text, vocab, model_path)
    print(result)
    
if __name__ == "__main__":
    # main()
    sample_test()