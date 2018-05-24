import pandas as pd
import numpy as np

trainFilePath1 = 'D:\\MyConfiguration\\szj46941\\Desktop\\beer标记过数据.csv'
trainFilePath2 = 'D:\\MyConfiguration\\szj46941\\Desktop\\beer标记过数据.excel'
stopWordsFilePath = 'D:\\MyConfiguration\\szj46941\\Desktop\\stopwords.txt'


def learnDf():
    # 属性、数据分开的列表
    # data = [['Alex',10],['Bob',12],['Clarke',13]]
    # columns = ['name','age']
    # df = pd.DataFrame(data,columns = columns,dtype=float,index=[3,4,5])
    # print(df)

    # json 属性列表
    # data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
    # df = pd.DataFrame(data,index=['rank1','rank2','rank3','rank4'])
    # print(df)

    # json 对象
    # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    # df = pd.DataFrame(data)
    # print(df)

    # Series指定数据和下标
    # data = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
    #         'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
    #
    # df = pd.DataFrame(data)
    # df['three'] = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    #
    # # 由内部数据生成列
    # df['four'] = df['three'] + df['two']
    # print(df)
    # 删除列
    # df.pop('four')
    # print(df)
    # 根据下标定位行
    # print(df.loc['a'])
    # 根据下标坐标定位行
    # print(df.iloc[2])
    # 行切片
    # print(df[0:3])
    # 附加行
    # df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    # df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])
    # df = df.append(df2)
    # # 删除行
    # print(df.drop(0))
    d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Minsu', 'Jack']),
         'Age': pd.Series([25, 26, 25, 23, 30, 29, 23]),
         'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8])}
    df = pd.DataFrame(d)
    print(df)
    print(df.T)
    print(df.axes)
    print(df.dtypes)
    print(df.ndim)
    print(df.size)
    print(df.shape)
    print(df.values)


def learnPanel():
    data = np.random.rand(2, 4, 5)
    p = pd.Panel(data)
    print(p)


def learnSeries():
    s = pd.Series(np.random.randn(2))
    # print(s.ndim)
    # print(s.empty)
    # print(s.axes)
    # print(s.values)
    # print(s.head(1))
    # print(s.tail(1))


def learnStatistics():
    # d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Minsu', 'Jack',
    #                         'Lee', 'David', 'Gasper', 'Betina', 'Andres']),
    #      'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]),
    #      'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65])}
    # df = pd.DataFrame(d)
    # print(df)
    # print(df.sum()) # print(df.sum(0))  axis=0 列
    # print(df.sum(1))
    # print(df.mean())
    # print(df.std()) # 标准差
    # print(df.describe())
    # s = pd.Series([1, 2, 3, 4, 5, 4])
    # print(s.pct_change())
    # df = pd.DataFrame(np.random.randn(5, 2))
    # print(df.pct_change()) # x上一个元素根当前元素的比
    # s1 = pd.Series(np.random.randn(10))
    # s2 = pd.Series(np.random.randn(10))
    # print(s1.cov(s2)) # 协方差
    s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))

    s['d'] = s['b']  # so there's a tie
    print(s)
    print(s.rank())  # 排名，最大的分最高


def learnApplication():
    def adder(ele1, ele2):
        return ele1 + ele2

    df = pd.DataFrame(np.random.randn(5, 3), columns=['col1', 'col2', 'col3'])
    # print(df)
    # print(df.pipe(adder, 2))
    # print(df.apply(lambda x: x.max() - x.min()))
    # print(df.mean())
    # print(df.apply(np.mean,axis=1))
    # print(df['col1'].map(lambda x:x*100))


def learnRebuildIndex():
    N = 20
    df = pd.DataFrame({
        'A': pd.date_range(start='2016-01-01', periods=N, freq='2D'),
        'x': np.linspace(0, stop=N - 1, num=N),
        'y': np.random.rand(N),
        'C': np.random.choice(['Low', 'Medium', 'High'], N).tolist(),
        'D': np.random.normal(100, 10, size=(N)).tolist()
    })
    print(df)
    # df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])
    # print(df_reindexed)
    #
    # df1 = pd.DataFrame(np.random.randn(10, 3), columns=['col1', 'col2', 'col3'])
    # df2 = pd.DataFrame(np.random.randn(7, 3), columns=['col1', 'col2', 'col3'])
    # df1 = df1.reindex_like(df2)
    # print(df1)
    # df2.reindex_like(df1, method='ffill',limit=1)
    # print(df1.rename(columns={'col1':'colx1'}))


def learnIteration():
    # important 迭代时无法修改值

    N = 20

    df = pd.DataFrame({
        'A': pd.date_range(start='2016-01-01', periods=N, freq='D'),
        'x': np.linspace(0, stop=N - 1, num=N),
        'y': np.random.rand(N),
        'C': np.random.choice(['Low', 'Medium', 'High'], N).tolist(),
        'D': np.random.normal(100, 10, size=(N)).tolist()
    })
    #
    # for col in df:
    #     print(col)
    # for k,v in df.iteritems():
    #     print(k,v)
    # for row_index,row in df.iterrows():
    #     print(row_index,row)
    # for row in df.itertuples():
    #     print(row)


def learnSort():
    unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])
    # print(unsorted_df)
    # sorted_df = unsorted_df.sort_index(ascending=False)
    # sorted_df = unsorted_df.sort_index(axis=1)
    sorted_df = unsorted_df.sort_values(by=['col1', 'col2'], kind='mergesort')

    print(sorted_df)


def learnStringAndText():
    s = pd.Series(['Tom', 'William Rick', 'John', 'John', 'Alber@t', np.nan, '1234', 'SteveMinsu'])
    print(s.str.lower())
    print(s.str.cat(sep=' <=> '))  # add separator
    print(s.str.get_dummies())  # 计算每个词在哪个下标出现
    print(s.str.contains('W'))


def learnOption():
    print("display.max_rows = ", pd.get_option("display.max_rows"))
    pd.set_option("display.max_rows", 80)
    print("after set display.max_rows = ", pd.get_option("display.max_rows"))
    pd.reset_option("display.max_rows")
    print("reset display.max_rows = ", pd.get_option("display.max_rows"))
    with pd.option_context("display.max_rows", 10):
        print(pd.get_option("display.max_rows"))
    print(pd.get_option("display.max_rows"))


def learnIndexAndSelectData():
    df = pd.DataFrame(np.random.randn(8, 4),
                      index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], columns=['A', 'B', 'C', 'D'])

    # Select few rows for multiple columns, say list[]
    print(df.loc[['a', 'b', 'f', 'h'], ['A', 'C']])
    # Select all rows for multiple columns, say list[]
    print(df.loc[:, ['A', 'C']])
    print(df.loc[['a'], ['A', 'C']])
    print(df.loc[['a', 'b', 'f', 'h'], ['A', 'C']])
    print(df.loc[['a'], :])
    print(df.iloc[:4])
    print(df.iloc[1:5, 2:4])  # 按行号
    print(df.loc['a'] > 0)  # 按行标签


def learnRolling():
    df = pd.DataFrame(np.random.randn(10, 4),
                      index=pd.date_range('1/1/2020', periods=10),
                      columns=['A', 'B', 'C', 'D'])
    print(df)
    print(df.rolling(window=3, min_periods=1).mean())  # 第三个元素的值将是n，n-1和n-2元素的平均值
    print(df.rolling(window=3, min_periods=2).mean())
    # print(df.expanding(min_periods=1).mean())
    r = df.rolling(window=3, min_periods=1)
    print(r['A'].aggregate([np.sum, np.mean]))


def learnLackData():
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
                                                    'h'], columns=['one', 'two', 'three'])
    df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    print(df)
    print(df.fillna(0))
    print(df.dropna())
    print(df.replace({1000: 10, 2000: 60}))


def learnGroup():
    ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
                         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
                'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
                'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
                'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
    df = pd.DataFrame(ipl_data)
    grouped = df.groupby('Year')
    print(grouped['Points'].agg(np.mean))  # 分组后聚合
    grouped = df.groupby('Team')
    score = lambda x: (x - x.mean()) / x.std() * 10
    print(grouped.transform(score))
    filter = df.groupby('Team').filter(lambda x: len(x) >= 3)  # 过滤
    print(filter)


def learnMergeAndJoin():
    left = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5']})
    right = pd.DataFrame(
        {'id': [1, 2, 3, 4, 5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5']})
    rs = pd.merge(left, right, on=['id', 'subject_id'], how='left') # left join
    print(rs)

def learnConcat():
    one = pd.DataFrame({
        'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5'],
        'Marks_scored': [98, 90, 87, 69, 78]},
        index=[1, 2, 3, 4, 5])
    two = pd.DataFrame({
        'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5'],
        'Marks_scored': [89, 80, 79, 97, 88]},
        index=[1, 2, 3, 4, 5])
    print(pd.concat([one, two], keys=['x', 'y'], ignore_index=True))
    print(one.append(two))

def learnTimeRange():
    time = pd.Timestamp('2018-11-01')
    print(time)
    time = pd.Timestamp(1588686880, unit='s')
    print(time)
    time = pd.date_range("12:00", "23:59", freq="30min").time
    print(time)
    time = pd.date_range("12:00", "23:59", freq="H").time # 偏移别名H
    print(time)
    time = pd.to_datetime(pd.Series(['Jul 31, 2009', '2019-10-10', None]))
    print(time)


def learnTimeDelta():
    timediff = pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
    print(timediff)
    s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
    td = pd.Series([pd.Timedelta(days=i) for i in range(3)])
    df = pd.DataFrame(dict(A=s, B=td))
    df['C'] = df['A'] + df['B']
    print(df)


def learnCategory():
    # s = pd.Series(["a", "b", "c", "a"], dtype="category")
    # print(s)
    # cat = cat=pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'],ordered=True)
    # print (cat)
    cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
    df = pd.DataFrame({"cat": cat, "s": ["a", "c", "c", np.nan]})
    print(df.describe())
    print("=============================")
    print(df["cat"].describe())

if __name__ == '__main__':
    learnCategory()
