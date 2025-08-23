+1 for each 8,9,6
–4 for each 4
+2 for every run of ≥3 identical digits (e.g. “1111”)
+2 for ascending runs of ≥4 (e.g. “1234”)
+2 for mirror/palindrome patterns (e.g. “1221”)



# 电话评分函数
def phone_scoring(phone):
    phone = phone[3:]
    n_4 = phone.count('4')
    n_6 = phone.count('6')
    n_8 = phone.count('8')
    n_9 = phone.count('9')
    if n_4 > 0:
        score = 0
    else:
        score = n_8 + n_9 + n_6 * 0.5

    return score

# 添加电话分数
def add_phone_score(df):
    df['phone_score'] = df['联系电话'].apply(lambda x: phone_scoring(str(x)))
    return df

# 添加手机号分数
    if 'phone_score' in df.columns:
        print('数据已包含手机号评分')
    else:
        print('开始手机号评分')
        df = add_phone_score(df)
        print('完成手机号评分')
