损失函数
function J = costFunctionJ(X,Y,theta)

%X is the 'design matrix' containing out training examples.
%Y is the class labels

predictions = X * theta;

e = predictions - y

J = e'*e/(2*m);
