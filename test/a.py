import numpy as np

if __name__ == '__main__':
    lines = open('C:\\Users\\admin\\Desktop\\beer标记过数据.csv').readlines()[:-1]
    print(lines[1])
    tags = ['ml' in line or 'ML' in line or '桶' in line or '听' in line or '箱' or '瓶' in line in line for line in lines]
    print(len(tags))
    print(tags.count(True))
    indexes = [i for i in range(len(tags)) if tags[i] == False]
    noises =[lines[x] +'--------------' for x in indexes]
    print(noises[:100])




