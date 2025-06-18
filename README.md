
## ems120 `v1.0`

`ems120` ，在 [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)的基础上，搭建疾病分类模型
<br>


## 1. Install 安装

<b>1.1</b> 下载基础包
```  
git clone https://https://github.com/jielab/ems120.git
cd ems120
``` 

<b>1.2</b> 安装Python依赖包
``` 
conda create -n ems120 python>3.11
conda activate ems120
pip install -r requirements.txt
``` 

<b>1.3</b> 下载 chinese-macbert-base 的核心文件 [pytorch_model.bin](https://huggingface.co/hfl/chinese-macbert-base/tree/main) ，放置于 hfl/chinese-macbert-base/ 文件夹，该文件包含预训练模型的所有参数。
<br><br>


## 2. Run 运行

<b>2.1</b>  训练模型，基于2020年的生成后的疾病分类训练的权重放置于 hfl 文件夹
```
python train_model.py
```

<b>2.2</b>  运行模型，对给定的每年的数据进行疾病分类
```
python ems-dx.py
```

<b>2.3</b>  基于“现场地址”，添加急救地点🚑的🗺坐标，示例数据 data/test.xlsx.
点击 [这儿](https://lbsyun.baidu.com)获取密钥，点击[这儿](https://lbsyun.baidu.com/cashier/quota)购买更多. 
```
python ems-map.py 
```



