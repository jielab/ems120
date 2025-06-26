
## ems120 🚑 急救120模型

`ems120` ，在 [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)的基础上，搭建疾病分类模型
<br><br>


## 1. Install 安装

<b>1.1</b> 下载基础包
```  
git clone https://github.com/jielab/ems120.git
cd ems120
``` 

<b>1.2</b> 安装Python依赖包
``` 
conda create -n py311 python=3.11
conda activate py311
pip install -r requirements.txt
``` 

<b>1.3</b> 下载 chinese-macbert-base 的核心文件 <b>[pytorch_model.bin](https://huggingface.co/hfl/chinese-macbert-base/tree/main)</b>
```  
放置于 hfl/chinese-macbert-base 文件夹，该文件包含预训练模型的所有参数。
```  
<br>


## 2. Run 运行

<b>2.1</b>  数据清洗。
```
python qc_data.py
```

<b>2.2</b>  训练模型。
```
python train_model.py
基于2019年数据的前1000条，进行训练，数据位于 data/2019.xlsx。
生成权重文件 trained_model.pth，放置于 hfl 文件夹。
```

<b>2.3</b>  运行模型。
```
python run_model.py
根据性别、年龄、主诉、现病史、初步诊断、补充诊断、呼救原因，进行疾病分类【一共25种疾病类型】。
```

<b>2.4</b>  添加坐标。
```
python add_xy.py 
示例数据位于 data/test.xlsx，基于文件的“现场地址”列。
从https://lbsyun.baidu.com获取密钥，从https://lbsyun.baidu.com/cashier/quota购买。
```



