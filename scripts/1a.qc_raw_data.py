import pandas as pd
from sympy.codegen.ast import continue_


#################################定义函数################################################################################

#######1.去掉无效记录
def clean(df):
    len1 = len(df.index)
    df=df[df['呼救类型'].isin(['疫情', '疫苗', '突发事件', '患者转院', '救治', '外院社康', '发热', '院内转院', '本院社康','市内转运(非急救)','传染病','转隔离点','一般公共事件', '中心用车','市外转运(非急救)'])]
    df.dropna(axis=0, subset=["接车地点", "驶向现场时刻"], how='any', inplace=True)
    df = df[~df.到达医院时刻.isnull()]
    df = df[df['是否正常结束']=='是']
    len2 = len(df.index)
    print('删除了' + str(len1 - len2) + '行无效记录')

    return df

#######2.添加时间差、手机号码分类、经纬度信息、地址类型、街道分类、疾病类型
# 添加时间差
def add_time(df):
    df['开始受理时刻'] = pd.to_datetime(df['开始受理时刻'])
    df['驶向现场时刻'] = pd.to_datetime(df['驶向现场时刻'])
    df['到达现场时刻'] = pd.to_datetime(df['到达现场时刻'])
    df['病人上车时刻'] = pd.to_datetime(df['病人上车时刻'])
    df['到达医院时刻'] = pd.to_datetime(df['到达医院时刻'])

    df['受理调度时间'] = (df['驶向现场时刻'] - df['开始受理时刻']).dt.seconds
    df['去程在途时间'] = (df['到达现场时刻'] - df['驶向现场时刻']).dt.seconds
    df['现场停车时间'] = (df['病人上车时刻'] - df['到达现场时刻']).dt.seconds
    df['返程在途时间'] = (df['到达医院时刻'] - df['病人上车时刻']).dt.seconds
    df['急救反应时间'] = df['受理调度时间'] + df['去程在途时间'] + df['现场停车时间'] + df['返程在途时间']

    return df


#############################################主函数######################################################################
if __name__ == "__main__":

    # 输入需要处理的数据路径，适用于2020年后的数据
    filepath = input('请输入需要QC的数据路径:')
    outputpath = input('请输入输出文件夹路径:')

    # 读取数据，原始数据前四行可能为数据介绍需要跳过
    try:
        df = pd.read_excel(filepath)
    except:
        df = pd.read_excel(filepath, skiprows=4)

    #备份原始文件
    if filepath==outputpath + '/' + filepath.split('/')[-1] or filepath==outputpath + filepath.split('/')[-1]:
        if bool(outputpath)==False:
            df.to_excel('origin_' + filepath.split('/')[-1])
        else:
            df.to_excel(outputpath+'/origin_' + filepath.split('/')[-1])

    # 去掉无效记录
    if df[df.到达医院时刻.isnull()].size==0:
        print('已去除过无效记录')
    else:
        print('开始去掉无效记录')
        df = clean(df)
        print('完成去掉无效记录')

    #添加时间差
    if '受理调度时间' in df.columns:
        print('数据已添加时间差')
    else:
        print('开始添加时间差')
        try:
            df = add_time(df)
        except:
            continue_
        print('完成时间差添加')
        

    df.to_excel(outputpath + '/processed_' + filepath.split('/')[-1])
    print('完成数据QC')
