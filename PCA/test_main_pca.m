clear all;
%% ������
tic
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% ��ά
K = 15;  %ά��
train = cell(10,1); 
train_norm = data_central(train_set); %���Ļ�����

[U, S] = eig((1/size(train_norm,1)) * train_norm' * train_norm);  %����ֵ�ֽ�
train_norm_project = projectData(train_set, U, K); %ӳ�䵽��Ӧά��

%% ��˹�ֲ���������
for i=1:10
    train{i,1} =  train_norm_project(find(train_label==i-1),:);
    mu(i,:) = mean_own(train{i,1}); %���ƾ�ֵ
    sigma(:,:,i) = (train{i,1}-mu(i,:))'*(train{i,1}-mu(i,:))/size(train{i,1},1);
    %����Э����
end

%% ��Ҷ˹����

test_norm_project = projectData(test_set, U, K);

for i = 1:10 %�������
    predic(i) = size(train{i,1},1)/size(train_set,1);
end

for i = 1:size(test_label)
    for j = 1:10
        gauss_bayes(i,j) = -0.5*(test_norm_project(i,:)-mu(j,:))*inv(sigma(:,:,j))*...
            (test_norm_project(i,:)-mu(j,:))'+log(predic(1,j))-0.5*log(det(sigma(:,:,j)));
        %��˹�ֲ�
        [a,result(i)] = max(gauss_bayes(i,:));
    end
end
toc
right = 1-size(find(result'-test_label-1~=0),1)/size(test_label,1);
