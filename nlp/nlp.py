import jieba


sent = '在包含问题的所有解的解空间树中，按照深度优先搜索的策略。从根节点出发深度搜索解空间树'

wlist = jieba.cut(sent,cut_all=True)

print("|".join(wlist))