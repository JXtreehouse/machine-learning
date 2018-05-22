import pandas as pd
import numpy as np

trainFilePath1 = 'D:\\MyConfiguration\\szj46941\\Desktop\\beer标记过数据.csv'
trainFilePath2 = 'D:\\MyConfiguration\\szj46941\\Desktop\\beer标记过数据.excel'
stopWordsFilePath = 'D:\\MyConfiguration\\szj46941\\Desktop\\stopwords.txt'


if __name__ == '__main__':
    # data = [['Alex',10],['Bob',12],['Clarke',13]]
    # columns = ['name','age']
    # df = pd.DataFrame(data,columns = columns,dtype=float,index=[3,4,5])
    # print(df)

    # data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
    # df = pd.DataFrame(data,index=['rank1','rank2','rank3','rank4'])
    # print(df)

    # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    # df = pd.DataFrame(data)
    # print(df)

    data = {'one':pd.Series([1,2,3],index=['a','b','c']),
            'two': pd.Series([1, 2, 3,4], index=['a', 'b', 'c','d'])}

    df = pd.DataFrame(data)
    df['three'] = pd.Series([1,2,3,4],index=['a','b','c','d'])
    df['four'] = df['three'] + df['two']
    print(df)