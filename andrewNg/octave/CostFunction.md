损失函数
function J = costFunctionJ(X,Y,theta)

%X is the 'design matrix' containing out training examples.
%Y is the class labels

predictions = X*theta;  %hypothesis 的值,theta和X样本的元素决定了h的值

sqrErrors = (predictions-Y).^2  %矩阵中元素平方

m = size(X,1)  %X矩阵的行数，也就是example的数量
J = 1/(2*m) * sum(sqrErrors);