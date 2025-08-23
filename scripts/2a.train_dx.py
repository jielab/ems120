import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer,AutoTokenizer,AutoModel,BertModel

# 模型定义
class Model(nn.Module):
    def __init__(self, CFG):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('./hfl/chinese-macbert-base')
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 14)  # 示例分类任务
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 25)  # 示例分类任务
    def forward(self, input_ids, attention_mask):
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        y1 = self.fc1(text)
        y2 = self.fc2(text)
        return y1, y2

# 数据集定义
class MyDataset(Dataset):
    def __init__(self, dataframe,tokenizer, CFG):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.CFG = CFG
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text = str(self.df.text.values[idx])
        label1 = int(self.df.label1.values[idx])
        label2 = int(self.df.label2.values[idx])
        # 使用 tokenizer 对文本进行编码
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.CFG['max_len'], return_tensors="pt")
        # 返回文本的编码（input_ids 和 attention_mask）以及标签
        input_ids = encoding['input_ids'].squeeze(0)  # 去除批次维度
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label1, label2

# 将数据以张量形式返回
def collate_fn(data):
    input_ids = torch.stack([x[0] for x in data])  # 将 input_ids 组成一个批次
    attention_mask = torch.stack([x[1] for x in data])  # 将 attention_mask 组成一个批次
    label1 = torch.LongTensor([x[2] for x in data])  # 将 label1 组成一个批次
    label2 = torch.LongTensor([x[3] for x in data])  # 将 label2 组成一个批次
    #print('1')
    return input_ids, attention_mask, label1, label2


# 配置参数
CFG = {
    'fold_num': 5, 'seed': 42, 'model': 'hfl/chinese-macbert-base', #中文预训练模型
    'max_len': 300, 'epochs': 10, 'train_bs': 32, 'valid_bs': 32, 'lr': 2e-5,
    'num_workers': 4, 'accum_iter': 1, 'weight_decay': 1e-6, 'device': 0,
}

# 加载数据
def get_loaders(dataframe, CFG):
    dataset = MyDataset(dataframe,tokenizer,CFG)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False)
    return train_loader, valid_loader

# 训练代码
def train_model(model, train_loader, valid_loader, optimizer, criterion, CFG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(CFG['epochs']):
        print(f"Epoch {epoch + 1}/{CFG['epochs']}")
        model.train()
        train_loss = 0
        i=0
        for batch in train_loader:
            optimizer.zero_grad()
            print(i)
            i=i+1
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            label1 = batch[2].to(device)
            label2 = batch[3].to(device)
            # Forward pass
            outputs1, outputs2 = model(input_ids, attention_mask)
            loss1 = criterion(outputs1, label1)
            loss2 = criterion(outputs2, label2)
            loss = loss1 + loss2
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        # Validation phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                label1 = batch[2].to(device)
                label2 = batch[3].to(device)
                outputs1, outputs2 = model(input_ids, attention_mask)
                loss1 = criterion(outputs1, label1)
                loss2 = criterion(outputs2, label2)
                loss = loss1 + loss2
                valid_loss += loss.item()
        print(f"Validation Loss: {valid_loss / len(valid_loader):.4f}")

# 主函数
if __name__ == "__main__":
    import pandas as pd
    # 模拟数据
    df=pd.read_excel('output/processed_2021.xlsx',usecols=['text','label1','label2'])
    df = pd.DataFrame(df)
    # 初始化模型、分词器、数据加载器
    tokenizer = BertTokenizer.from_pretrained('./hfl/chinese-macbert-base')
    model = Model(CFG)
    train_loader, valid_loader = get_loaders(df,  CFG)
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = nn.CrossEntropyLoss()
    # 开始训练
    train_model(model, train_loader, valid_loader, optimizer, criterion, CFG)
    torch.save(model.state_dict(), 'hfl/trained_model.pth')
