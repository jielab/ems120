
## ems120 🚑


## 1. Install

<b>1.1</b> Install basic version
```  
git clone https://github.com/jielab/ems120.git
cd ems120
``` 

<b>1.2</b> Install Python dependencies
``` 
conda create -n py311 python=3.11
conda activate py311
pip install -r requirements.txt

If the above does not work on HPC, install separately:
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers pandas openpyxl
``` 

<b>1.3</b> Download chinese-macbert-base modeling [parameter file](https://huggingface.co/hfl/chinese-macbert-base/tree/main)
```  
After "git clone", manually download pytorch_model.bin and put into hfl/chinese-macbert-base/
```  
<br>


## 2. Run

<b>2.1</b> QC data & Add geographic info
```
python 1a.qc_data.py
python 1b.add_geo.py # test data/geo.test.xlsx 
# obtain key from https://lbsyun.baidu.com[/cashier/quota]
```

<b>2.2</b> Add Dx
```
python 2a.train_dx.py # input: data/dx.test.xlsx; output: param/trained_dx.pth
python 2b.add_dx.py
```

<b>2.3</b> Add phone luckiness
```
python 3a.train_phone.py # input: data/luck.test.xlsx; output: param/train_luck.pth
python 3b.add_luck.py
```




