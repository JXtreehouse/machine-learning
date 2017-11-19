 t=[0:0.01:1] %生成间隔0.01,从0到1的一个矩阵
 y1=sin(2*pi*4*t) %周期0.25的正弦函数
 y2=cos(8*pi*t)
 plot(t,y1) %生成图像
 hold on; %保留图像
 plot(t,y2) %生成图像
 title plot %图像标题
 ylabel('value') %标记y轴
 xlabel('time') %标记x轴
 legend('sin','cos') %标记两条曲线
 print -dpng 'myFirtPlot.png' %生成图片