clear all;
clc;
%% 读入数据
% data = load('data0.txt');
% data = load('data1.txt');
% data = load('data2.txt');
% data = load('data3.txt');
data = load('data4.txt');

len = size(data,1);
x = data(:,1);
y = data(:,2);

%% 初始化
e = 0.05;
iter = 100; % 迭代次数
rou = 1;   %惩罚系数
c = 1;
lamna = 0.1*ones(4*len,1);  %拉格朗日乘子
% 将矩阵不等式变换成为Ax<=b
A1  = [-2*x,ones(len,1),-2*y,ones(len,1),-ones(len,1),-eye(len),zeros(len,len)];
A2  = [2*x,-ones(len,1),2*y,-ones(len,1),ones(len,1),zeros(len,len),-eye(len) ];
A3 = [zeros(len,5),-eye(len), zeros(len,len)];
A4 = [zeros(len,5),zeros(len,len), -eye(len)];
A = [A1;A2;A3;A4];
len_d = size(A,1);
b = [- x.^2 - y.^2 + e; x.^2 + y.^2 - e; zeros(len,1); zeros(len,1)];
    % a,A,b,B,R    
f = [0,0,0,0,0,c*ones(1,len),c*ones(1,len)]; %目标函数为f*x

%% 对偶下降法应用，采用增广的目标函数
for i = 1:iter
%     argmin_x =  pinv(rou*A'*lamna*lamna'*A)*(rou*A'*lamna*lamna'*b-A'*lamna-f');
    argmin_x =  pinv(rou*A'*A)*(rou*A'*b-A'*lamna-f');
        %对x求最小
    lamna = lamna + rou*(A*argmin_x-b);
        %对拉格朗日乘子进行迭代
    for j = 1:len_d  %拉格朗日乘子必须大于等于零
        if lamna(j)<0
            lamna(j) = 0;
        end
    end
end

[aa,AA,bb,BB,rr ] = deal(argmin_x(1),argmin_x(2),argmin_x(3),argmin_x(4),argmin_x(5));
[xx_l,yy_l] = linprog(f',A,b,[],[],[]); %对曲线进行线性拟合，结果作为标准值
[xx_c,yy_c,r_c] = circfit(x,y);

%还原得到半径的值
r = sqrt(e + rr - AA - BB + aa^2 + bb^2);
r_ling = sqrt(e + xx_l(5) - xx_l(4) - xx_l(2) + xx_l(1)^2 +xx_l(3)^2 );

%% 绘出结果图
theta = 0:0.01:2*pi+0.1;  
Circle_x1 = aa+r*cos(theta);  Circle_y1 = bb+r*sin(theta);  
Circle_x2 = xx_l(1)+r_ling*cos(theta);  Circle_y2 = xx_l(3)+r_ling*sin(theta);
Circle_x3 = xx_c+r_c*cos(theta);  Circle_y3 = yy_c+r_c*sin(theta); 
scatter(aa,bb,'r','linewidth',3);
hold on
plot(Circle_x1,Circle_y1,'r','linewidth',3); 
scatter(xx_l(1),xx_l(3),'g','linewidth',1);
plot(Circle_x2,Circle_y2,'g','linewidth',1);
scatter(xx_c,yy_c,'b','linewidth',1);
plot(Circle_x3,Circle_y3,'b','linewidth',1);
scatter(x,y,'linewidth',1);
legend('svr圆心','svr拟合','线性拟合圆心','线性拟合','最小二乘拟合圆心','最小二乘拟合','数据点');
axis equal 
% [xx,yy] = linprog(f',A,b,[],[],[]);

