
## 🚑 ems120 


## 1. Install

<b>1.1</b> Install basic version
```  
git clone https://github.com/jielab/ems120.git
cd ems120
``` 

<b>1.2</b> Install Python dependencies
``` 
> conda create -n py311 python=3.11
> conda activate py311
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers pandas openpyxl numpy tqdm requests
``` 

<b>1.3</b> Download chinese-macbert-base modeling [parameter file](https://huggingface.co/hfl/chinese-macbert-base/tree/main)
```  
save to a directory that will be called as "model_bert" in python.
```  
<br>


## 2. Run

<b>2.1</b> ⛑ Add Dx
```
python ems120_dx.py
```

<b>2.2</b> 📱 Add phone luckiness
```
python ems120_luck.py
```

<b>2.3</b> 🏠 Add geographic & housing price
```
python ems120_geo.py # key from https://lbsyun.baidu.com[/cashier/quota]
```