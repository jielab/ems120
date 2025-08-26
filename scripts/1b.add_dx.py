import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import os
import pandas as pd

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, CFG):
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


# 定义基于预训练BBERT的分类模型
class Model(nn.Module):
    def __init__(self, CFG):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained('../hfl/chinese-macbert-base')
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 14)  # 示例分类任务
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 25)  # 示例分类任务
    def forward(self, input_ids, attention_mask):
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        y1 = self.fc1(text)
        y2 = self.fc2(text)
        return y1, y2

# 将数据以张量形式返回
def collate_fn(data):
    input_ids = torch.stack([x[0] for x in data])  # 将 input_ids 组成一个批次
    attention_mask = torch.stack([x[1] for x in data])  # 将 attention_mask 组成一个批次
    label1 = torch.LongTensor([x[2] for x in data])  # 将 label1 组成一个批次
    label2 = torch.LongTensor([x[3] for x in data])  # 将 label2 组成一个批次
    return input_ids, attention_mask, label1, label2
def classify_disease(df):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    df['label1'] = 0
    df['label2'] = 0
    df['text'] = df['性别'].astype(str) + '[SEP]' + df['年龄'].astype(str) + '[SEP]' + \
                 df['主诉'].astype(str) + '[SEP]' + df['现病史'].astype(str) + '[SEP]' + \
                 df['初步诊断'].astype(str) + '[SEP]' + df['补充诊断'].astype(str) + df['呼救原因'].astype(str)
    CFG = {
        'fold_num': 5, 'seed': 42, 'model': 'hfl/chinese-macbert-base',  # 中文预训练模型
        'max_len': 300, 'epochs': 10, 'train_bs': 32, 'valid_bs': 32, 'lr': 2e-5,
        'num_workers': 4, 'accum_iter': 1,  'weight_decay': 1e-6, 'device': device,
    }

    labels1 = ['儿科', '其他', '内分泌系统疾病', '创伤', '呼吸系统疾病', '妇产科', '心脏骤停', '心血管系统疾病',
               '感染性疾病', '泌尿系统疾病', '消化系统疾病', '理化中毒', '神经系统疾病', '精神病']
    labels2 = ['儿科','其他-休克', '其他-其他症状', '其他-意识不清', '其他-昏迷', '其他-死亡', '其他-胸闷', '内分泌系统疾病',
               '创伤-交通事故', '创伤-其他原因', '创伤-暴力事件', '创伤-跌倒', '创伤-高处坠落', '呼吸系统疾病', '妇产科',
               '心脏骤停', '心血管系统疾病-其他疾病', '心血管系统疾病-胸痛', '感染性疾病', '泌尿系统疾病', '消化系统疾病',
               '理化中毒', '神经系统疾病-其他疾病', '神经系统疾病-脑卒中', '精神病']

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained('../hfl/chinese-macbert-base')
    # 定义数据集
    test_set = MyDataset(df,tokenizer,CFG)
    # 创建测试数据加载器
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=4)

    # 加载预训练模型
    print('加载预训练模型')
    model = Model(CFG).to(device)
    print('加载权重文件')
    model.load_state_dict(torch.load('../hfl/trained_model.pth', map_location=device, weights_only=True))
    model.eval()

    # 初始化预测过程
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pred = []
    print('开始预测')
    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            input_ids, attention_mask, y1, y2 = [x.to(device) for x in batch]
            _, output = model(input_ids, attention_mask)
            output = output.softmax(1).cpu().numpy()
            pred.extend(output)
    pred = np.array(pred)
    df['disease_label'] = pred.argmax(1)
    df['疾病分类'] = df['disease_label'].apply(lambda x: labels2[x])
    df=df.drop(columns=['label1','label2','text','disease_label'])
    #print('完成')
    return df

if __name__ == "__main__":
    # 输入需要处理的数据路径，适用于2020年后的数据
    filepath = input('请输入需要分类的数据路径:')
    outputpath = input('请输入输出文件夹路径:')
    # 读取数据，原始数据前四行可能为数据介绍需要跳过
    try:
        df = pd.read_excel(filepath)
    except:
        df = pd.read_excel(filepath, skiprows=4)
    # 添加疾病分类
    if '疾病分类' in df.columns:
        print('数据已包含疾病分类')
    else:
        print('开始疾病分类')
        df = classify_disease(df)
        print('完成疾病分类')
    df.to_excel(outputpath + '/processed_' + filepath.split('/')[-1])
