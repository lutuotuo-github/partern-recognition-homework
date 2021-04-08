clear all;
clc;
%% ��������
% data = load('data0.txt');
% data = load('data1.txt');
% data = load('data2.txt');
% data = load('data3.txt');
data = load('data4.txt');

len_d = size(data,1);
x = data(:,1);
y = data(:,2);

%% ��ʼ��
e = 0.0005;
iter = 100; % ��������
t = 100;   %�ͷ�ϵ��
c = 0.1;
lamna = 0.1*ones(4*len_d,1);  %�������ճ���
v = ones(4*len_d,1);  %�ɳ����� ���붼������
arg_x = ones(5+2*len_d,1);
% �����󲻵�ʽ�任��ΪAx+b-v=0
A1  = [-2*x,ones(len_d,1),-2*y,ones(len_d,1),-ones(len_d,1),-eye(len_d),zeros(len_d,len_d)];
A2  = [2*x,-ones(len_d,1),2*y,-ones(len_d,1),ones(len_d,1),zeros(len_d,len_d),-eye(len_d) ];
A3 = [zeros(len_d,5),-eye(len_d), zeros(len_d,len_d)];
A4 = [zeros(len_d,5),zeros(len_d,len_d), -eye(len_d)];
A = -[A1;A2;A3;A4];
len_x = size(A,2);
b = [- x.^2 - y.^2 + e; x.^2 + y.^2 - e; zeros(len_d,1); zeros(len_d,1)];
% b = [- x.^2 - y.^2 + e; x.^2 + y.^2 + e; zeros(len_d,1); zeros(len_d,1)];
    % a,A,b,B,R    
f = [0,0,0,0,0,c*ones(1,len_d),c*ones(1,len_d)]; %Ŀ�꺯��Ϊf*x

%% ��ż�½���Ӧ�ã����������Ŀ�꺯��
for i = 1:iter
    arg_x =  pinv(t*A'*A)*(-t*A'*(b-v)+A'*lamna-f');
    mid1 = A*arg_x+b-lamna/t;
    for m = 1:len_x
        if mid1(m)>=0
            v(m) = mid1(m);
        else
            v(m) = 0;
        end
    end
    mid2 = lamna-t*(A*arg_x+b-v);
    for n = 1:len_x
        if mid2(n)>=0
            lamna(n) = mid2(n);
        else
            lamna(n) = 0;
        end
    end
end

[aa,AA,bb,BB,rr ] = deal(arg_x(1,1),arg_x(2,1),arg_x(3,1),arg_x(4,1),arg_x(5,1));
[xx_l,yy_l] = linprog(f',-A,b,[],[],[]); %�����߽���������ϣ������Ϊ��׼ֵ
[xx_c,yy_c,r_c] = circfit(x,y);

%��ԭ�õ��뾶��ֵ
r = sqrt(e + rr - AA - BB + aa^2 + bb^2);
% r = sqrt(rr);
r_liner = sqrt(e + xx_l(5) - xx_l(4) - xx_l(2) + xx_l(1)^2 +xx_l(3)^2 );
% r_ling = sqrt(xx_l(5));
%% ������ͼ
theta = 0:0.01:2*pi+0.1;  
Circle_x1 = aa+r*cos(theta);  Circle_y1 = bb+r*sin(theta);  
Circle_x2 = xx_l(1)+r_liner*cos(theta);  Circle_y2 = xx_l(3)+r_liner*sin(theta);
Circle_x3 = xx_c+r_c*cos(theta);  Circle_y3 = yy_c+r_c*sin(theta); 
scatter(aa,bb,'r','linewidth',3);

hold on
plot(Circle_x1,Circle_y1,'r','linewidth',3); 
scatter(xx_l(1),xx_l(3),'g','linewidth',1);
plot(Circle_x2,Circle_y2,'g','linewidth',1);
scatter(xx_c,yy_c,'b','linewidth',1);
plot(Circle_x3,Circle_y3,'b','linewidth',1);
scatter(x,y,'linewidth',1);
legend('svrԲ��','svr���','�������Բ��','�������','��С�������Բ��','��С�������','���ݵ�');
axis equal 


% [xx,yy] = linprog(f',A,b,[],[],[]);

