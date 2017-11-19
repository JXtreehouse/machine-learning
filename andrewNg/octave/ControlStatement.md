v=zeros(10,1);
for i=1:10
 v=2^i
end %把v的每个元素替换成2^i

indices=1:10
for i=indices %同样的遍历

i=1
while i<=5,
 v(i)=100;
 i=i+1
end;
%while和for可以用break。
%控制语句必须end结束


if xxx,
elseif xxx
else xxx
end

exit to exit octave