clc;
clear all;

imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 10;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

train_images = loadMNISTImages('train-images-idx3-ubyte');
% K = 144;  %维度
% [train_images,Utrain] = pca_own(trainimages',K);%进行pca降维

train_images = reshape(train_images,imageDim,imageDim,[]);
trainlabels = loadMNISTLabels('train-labels-idx1-ubyte');
trainlabels(trainlabels==0) = 10; % Remap 0 to 10
test_Images = loadMNISTImages('t10k-images-idx3-ubyte');

% test_Images =  projectData(testImages', Utrain, K);%用相同的转换矩阵进行投影
test_Images = reshape(test_Images,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; 

% K = 60;  %维度
% [train_project,Utrain] = pca_own(trainimages,K);%进行pca降维
% test_project =  projectData(test_set, Utrain, K);%用相同的转换矩阵进行投影

activationType = 'relu';
%'relu'
%'sigmoid'
epochs = 1;
minibatch = 200;
alpha = 0.1;
mom = 0.9;
numImages = length(trainlabels);

% theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);
w_filters = 1e-1*randn(filterDim,filterDim,numFilters);
outDim = imageDim - filterDim + 1; 
outDim = outDim/poolDim;
hiddennet = outDim^2*numFilters;
r  = sqrt(6)/sqrt(numClasses+hiddennet+1);
w_net = rand(numClasses, hiddennet) * 2*r  - r;

theta = [w_filters(:) ; w_net(:)];

velocity = zeros(size(theta));
%% 带有动量的Mini-batch gradient descent 小批量梯度下降
batch_num = 0;
tic
right = zeros(minibatch+1,1);
for i = 1:epochs
    rand_pic = randperm(numImages);%每次训练随机打乱图片顺序
    for j=1:minibatch:(numImages-minibatch+1)
        batch_num = batch_num + 1;
        minibatch_data = train_images(:,:,rand_pic(j:j+minibatch-1));
        minibatch_labels = trainlabels(rand_pic(j:j+minibatch-1));
        [cost, grad] = cnn_train(theta,minibatch_data,minibatch_labels,numClasses,filterDim,numFilters,poolDim,activationType,false);
        velocity = mom * velocity + alpha .* grad;
        
        theta = theta - velocity;

%         theta = theta + alpha .* grad;
        fprintf('Epoch %d/%d: Cost on batch %d is %f\n',i,epochs,batch_num,cost);
    end
    alpha = alpha/2.0;
    batch_num = 0;
end
toc
[~,~,preds] = cnn_train(theta,test_Images,testLabels,numClasses,filterDim,numFilters,poolDim,activationType,true);
right = sum(preds==testLabels)/length(preds);

fprintf('Accuracy is %f\n',right);
