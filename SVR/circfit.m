function [xc,yc,r] = circfit(x,y)
%圆拟合函数
% x^2+y^2+a(1)*x+a(2)*y+a(3)=0

    n=length(x); xx=x.*x; yy=y.*y; xy=x.*y;
    A=[sum(x) sum(y) n; sum(xy) sum(yy) sum(y);sum(xx) sum(xy) sum(x)];
    B=[-sum(xx+yy) ; -sum(xx.*y+yy.*y) ; -sum(xx.*x+xy.*y)];
    a=A\B;            %x = A\B 用来求解线性方程 A*x = B.  A 和 B 的行数一致.
    xc = -.5*a(1);
    yc = -.5*a(2);
    r = sqrt((a(1)^2+a(2)^2)/4-a(3));
   
%     theta=0:0.1:2*pi+0.1;  
%         Circle1=xc+R*cos(theta);  
%         Circle2=yc+R*sin(theta);   
%     plot(Circle1,Circle2,'g','linewidth',1);  
%     axis equal  
end 
