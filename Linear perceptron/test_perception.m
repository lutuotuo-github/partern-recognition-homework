clear all;
%% ����Ԥ����
tic
train_set = loadMNISTImages('train-images.idx3-ubyte')';%��ȡ����
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

K = 60;  %ά��
[train_project,Utrain] = pca_own(train_set,K);%����pca��ά
test_project =  projectData(test_set, Utrain, K);%����ͬ��ת���������ͶӰ
train = cell(10,1); 
test = cell(10,1);
reb_train = cell(10,1);
reb_test = cell(10,1);
for i=1:10
    train{i,1} =  train_project(find(train_label==i-1),:);
    test{i,1} =  test_project(find(test_label==i-1),:);
    reb_train{i,1} = [ones(size(train{i,1},1),1),train{i,1}];%������������
    reb_test{i,1} = [ones(size(test{i,1},1),1),test{i,1}];
end
uni_train = cell(10,10);
uni_test = cell(10,10);
for i = 1:10
    for j = 1:10
        uni_train{i,j} = [reb_train{i,1}(1:1000,:)',- reb_train{j,1}(1:1000,:)']; 
        %ѡ���1000�����������������
        uni_test{i,j} = [reb_test{i,1}',-reb_test{j,1}'];
    end
end
toc

err = zeros(10,10);
num = zeros(10,10);


%% ���Ը�֪����ѵ��
tic
eta = 0.01;%�ɵ�����
count = zeros(10,10);%��Ӧ�ĵ�������
lost_Ja = zeros(10,10);%��ʧ����
a = ones(K+1,1);
b = 2; %����
for i = 1:9
    for j = (i+1):10
        while(size(find(sign(a'*uni_train{i,j})<0),2)~=0)
            %����ȫ�ɷ����,�����������Լ������ֹͣ���� 
            lost_Ja_last = lost_Ja(i,j);
            for t = 1:2000
                test_step = a'*uni_train{i,j}(:,t)-b;%�ж�һ��
                if test_step <= 0
%                     a = a - abs(test_step)*(a'*uni_train{i,j}(:,t))*uni_train{i,j}(:,t)/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t))^2;
                    %�����ᵽ�ļ��ٵ��������Ĳ�������
%                     a = a + eta*uni_train{i,j}(:,t);
                    %һ����ʧ�����ĵ���
                    a = a - eta*(test_step)*uni_train{i,j}(:,t)/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t));
                     %������ʧ�����ĵ���
                    lost_Ja(i,j) = lost_Ja(i,j) + test_step^2/(uni_train{i,j}(:,t)'*uni_train{i,j}(:,t));
                    %������ʧ���������
%                     lost_Ja(i,j) = lost_Ja(i,j) + test_step;
                    %һ����ʧ���������
                    count(i,j) = count(i,j) + 1;%������������
                end
                
            end   
            if (lost_Ja_last - lost_Ja(i,j))^2/lost_Ja_last^2 < 1e-6
                %����ʧ��������Լ��������仯��С�������1e-6ʱ��ֹͣѭ�����õ�Ȩ������
                break;
            end
        end
        err(i,j) = size(find(sign(a'*uni_test{i,j}) == -1),2)/size(sign(a'*uni_test{i,j}),2);
        %�����ʵļ���
        a = ones(K+1,1);
    end
end
right = ones(10,10);
right = 1-err;%ת������ȷ��
toc




