v=zeros(10,1) %生成一个10行一列的矩阵

x = [1 1; 1 2; 1 3;]%生成一个3行2列的矩阵， ';'代表换行

addpath('C:\Desktop"') %把桌面设置成octave的运行环境

function y = squareThisNumber(x)
y=x^2%函数体

function[a,b] = squareAndCube(x)
a=x^2
b=x^3

X=data[:,1] %获取MxN矩阵的第一列
