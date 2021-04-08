clear all;
%% 数据预处理
tic
train_set = loadMNISTImages('train-images.idx3-ubyte')';%读取数据
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

K = 60;  %维度
[train_project,Utrain] = pca_own(train_set,K);%进行pca降维
test_project =  projectData(test_set, Utrain, K);%用相同的转换矩阵进行投影
train = cell(10,1); 
test = cell(10,1);
reb_train = cell(10,1);
reb_test = cell(10,1);
for i=1:10
    train{i,1} =  train_project(find(train_label==i-1),:);
    test{i,1} =  test_project(find(test_label==i-1),:);
    reb_train{i,1} = [ones(size(train{i,1},1),1),train{i,1}];%增广样本矩阵
    reb_test{i,1} = [ones(size(test{i,1},1),1),test{i,1}];
end
uni_train = cell(10,10);
uni_test = cell(10,10);
for i = 1:10
    for j = 1:10
        uni_train{i,j} = [reb_train{i,1}(1:1000,:)',- reb_train{j,1}(1:1000,:)']; 
        %选择各1000个正负样本进行组合
        uni_test{i,j} = [reb_test{i,1}',-reb_test{j,1}'];
    end
end
toc

err = zeros(10,10);
num = zeros(10,10);


%% 线性感知器的训练
tic
eta = 0.01;%可调步长
count = zeros(10,10);%对应的迭代次数
lost_Ja = zeros(10,10);%损失函数
a = ones(K+1,1);
b = 2; %余量
for i = 1:9
    for j = (i+1):10
        while(size(find(sign(a'*uni_train{i,j})<0),2)~=0)
            %若完全可分最好,若不能则加入约束条件停止迭代 
            lost_Ja_last = lost_Ja(i,j);
            for t = 1:2000
                test_step = a'*uni_train{i,j}(:,t)-b;%判定一次
                if test_step <= 0
%                     a = a - abs(test_step)*(a'*uni_train{i,j}(:,t))*uni_train{i,j}(:,t)/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t))^2;
                    %书中提到的减少迭代次数的步长方法
%                     a = a + eta*uni_train{i,j}(:,t);
                    %一次损失函数的迭代
                    a = a - eta*(test_step)*uni_train{i,j}(:,t)/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t));
                     %二次损失函数的迭代
                    lost_Ja(i,j) = lost_Ja(i,j) + test_step^2/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t));
                    %二次损失函数的求和
%                     lost_Ja(i,j) = lost_Ja(i,j) + test_step;
                    %一次损失函数的求和
                    count(i,j) = count(i,j) + 1;%迭代次数计算
                end
                
            end   
            if (lost_Ja_last - lost_Ja(i,j))^2/lost_Ja_last^2 < 1e-6
                %对损失函数进行约束，当其变化量小于自身的1e-6时，停止循环，得到权重向量
                break;
            end
        end
        err(i,j) = size(find(sign(a'*uni_test{i,j}) == -1),2)/size(sign(a'*uni_test{i,j}),2);
        %错误率的计算
        a = ones(K+1,1);
    end
end
right = ones(10,10);
right = 1-err;%转换成正确率
toc




