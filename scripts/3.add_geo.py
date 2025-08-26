import requests
import math
import pandas as pd


# 地理编码
def encoding(address, key):
    # 接口地址
    url = "https://api.map.baidu.com/geocoding/v3"

    # 此处填写你在控制台-应用管理-创建应用后获取的AK
    params = {"address": address, "output": "json", "ak": key, "city": "深圳市"}
    response = requests.get(url=url, params=params)
    if response:
        return response.json()
    else:
        print('地址', address, '  地理编码失败')

# 坐标系转换
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率

def gcj02towgs84(lng, lat):
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]

def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret

# 添加经纬度，地址类型
def add_xy(df,key):
    print('还有' + str(len(df[df['现场地址纬度'] == 0])) + '条记录待添加经纬度')
    for i in df.index:
        if df.loc[i, '现场地址纬度'] == 0:
            # print(i)
            xy_info = encoding(df.loc[i, '现场地址'], key)
            if xy_info['status'] == 0:
                df.loc[i, '现场地址纬度_原始'] = xy_info['result']['location']['lat']
                df.loc[i, '现场地址经度_原始'] = xy_info['result']['location']['lng']
                lng, lat = gcj02towgs84(xy_info['result']['location']['lng'], xy_info['result']['location']['lat'])
                df.loc[i, '现场地址纬度'] = lat
                df.loc[i, '现场地址经度'] = lng
                df.loc[i, '地址类型'] = xy_info['result']['level']
            elif xy_info['status'] == 2 or xy_info['status'] == 1:
                # print("跳过")
                df.loc[i, '现场地址纬度_原始'] = -1
                df.loc[i, '现场地址经度_原始'] = -1
                df.loc[i, '现场地址纬度'] = -1
                df.loc[i, '现场地址经度'] = -1
                continue
            else:
                print('密钥错误或已达到限额，请切换密钥或明日继续')
                break

    if len(df[df['现场地址经度'] == 0]) == 0:
        print('地理编码全部完成')

    return df


if __name__ == "__main__":
    filepath = input('请输入需要地理编码的数据路径:')
    outputpath = input('请输入输出文件夹路径:')
    key = input('请输入您的百度地图API密钥:')
    # 读取数据，原始数据前四行可能为数据介绍需要跳过
    try:
        df = pd.read_excel(filepath)
    except:
        df = pd.read_excel(filepath, skiprows=4)
    # 添加经纬度与地址类型
    if '现场地址经度' not in df.columns:
        print('开始添加经纬度与地址类型')
        df['现场地址经度'] = 0
        df['现场地址纬度'] = 0
        df = add_xy(df,key)
    while len(df[df['现场地址经度'] == 0]) != 0:
        dd = input('是否切换密钥继续(是/否):')
        if dd == '是':
            key = input('请输入您的百度地图API密钥:')
            add_xy(df,key)
        else:
            break

    df.to_excel(outputpath + '/Geocoded_' + filepath.split('/')[-1])

