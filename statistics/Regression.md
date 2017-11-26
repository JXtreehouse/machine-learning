梯度下降或者normal equation 来计算theta
找到costJ最小的theta

1. Linear Regression 预测 predict
hypothesis = h(theta转置 乘 f(特征值))

2 .Logistic Regression 分类 classify
a=theta转置 乘 f(特征值)
hypothesis = sigmoid function(a)
a<0时 hypothesis <0.5  -> y=0
a>0时 hypothesis >=0.5 -> y=1
