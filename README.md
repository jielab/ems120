
## ems120 `v1.0`

`ems120` is a command line tool for QC ems data

## 1. Getting Started

Download the basic files
```  
git clone https://https://github.com/jielab/ems120.git
cd ems120
``` 

Install the Python dependencies
``` 
conda create -n ems120 python>3.11
conda activate ems120
pip install -r requirements.txt
``` 

## 2. Download supporting data for test-run
> - Download [training weight files](https://www.abc.com), put into hfl/ folder.
> - Download [Macbert pretrained pytorch_model.bin file](https://www.abc.com), put into hfl/chinese-macbert-base/ folder.
> - Make sure test data has columns such as '现场地址', as shown in data/test.xlsx file.
> - Obtain Baidu map API key from [here](https://lbsyun.baidu.com). The limit is mapping 5000 addresses per day. More can be bought from [here](https://lbsyun.baidu.com/cashier/quota).
``` 
Ev5QMk3??DFRrJ?cfBznvzJnxq1OiSYi
``` 

## 3. Test-run
```
python scripts/ems-dx.py
python scripts/ems-map.py
```



