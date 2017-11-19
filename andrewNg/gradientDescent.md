梯度下降法：让costFunctionJ的值收敛到0
原理是让J的（theta1，theta2）往梯度反方向移动，这样J就会收敛到0。

梯度反方向为什么是函数值下降最快的方向？
因为方向导数D=x的偏导*cosθ+y的偏导*sinθ，
向量a:（x偏导，y偏导）和b:（cosθ，sinθ）内积就是D=|a|*|1|cosα，而梯度的模就是|a|，
那么D最小时应该是当α取π时，也就是和方向和梯度相反的方向（梯度方向是固定的，cosα只和方向导数单位向量方向有关，D=grad*el=|a|*|1|cosα）。

value scaling 特征值除以特征范围
mean normalization 特征值减去平均特征值

normal equation 方法求h，应该是求costFunctionJ的最小值时的theta