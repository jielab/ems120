
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

<b>1.3</b> Download chinese-macbert-base modeling parameter files <b>[pytorch_model.bin](https://huggingface.co/hfl/chinese-macbert-base/tree/main)</b>
```  
put under hfl/chinese-macbert-base。
```  
<br>


## 2. Run

<b>2.1</b>  QC & add geographic info
```
python 1a.qc_raw_data.py
python 1b.get_geo_loc.py # based on column 现场地址, example at data/test.xlsx. 
# obtain key from https://lbsyun.baidu.com，https://lbsyun.baidu.com/cashier/quota.
```

<b>2.2</b>  Add Dx
```
python 2a.train_dx.py # INPUT: 1,000 records from data/2019.xlxs; output: hfl/trained_dx.pth
python 2b.get_dx.py # based on 性别、年龄、主诉、现病史、初步诊断、补充诊断、呼救原因
```

<b>2.3</b>  Add phone luckiness
```
python 3a.train_phone_sco.py
python 3b.get_phone_sco.py
```




