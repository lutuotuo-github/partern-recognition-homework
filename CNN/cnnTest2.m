clc;
clear all;
%% 读入数据
cnn.imageDim = 28;
cnn.numClasses = 10; 
trainimages = loadMNISTImages('train-images-idx3-ubyte');
trainimages = reshape(trainimages,cnn.imageDim,cnn.imageDim,[]);
trainlabels = loadMNISTLabels('train-labels-idx1-ubyte');
trainlabels(trainlabels==0) = 10; 
testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,cnn.imageDim,cnn.imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; 
%% 数据初始化
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 10, 'kernelsize', 9) 
    struct('type', 's', 'scale', 2) 
};
cnn = cnnsetup2(cnn, trainimages);
cnn.activationType = 'relu';
%'relu'
%'sigmoid'

cnn.epochs = 1;
cnn.minibatch = 200;
cnn.alpha = 0.1;
cnn.mom = 0.9;
cnn.numImages = length(trainlabels);
velocity = zeros(size(cnn.theta));

%% 带有动量的Mini-batch gradient descent 小批量梯度下降
batch_num = 0;
tic
for i = 1:cnn.epochs
    rand_pic = randperm(cnn.numImages);%每次训练随机打乱图片顺序
    for j=1:cnn.minibatch:(cnn.numImages-cnn.minibatch+1)
        batch_num = batch_num + 1;
        minibatch_data = trainimages(:,:,rand_pic(j:j+cnn.minibatch-1));
        minibatch_labels = trainlabels(rand_pic(j:j+cnn.minibatch-1));

        [cost, grad] = cnn_train2(cnn,minibatch_data,minibatch_labels,false);
        velocity = cnn.mom * velocity + cnn.alpha .* grad;% 带有动量
        cnn.theta = cnn.theta - velocity;
%         cnn.theta = cnn.theta - cnn.alpha .* grad; % 批量梯度下降
        fprintf('Epoch %d/%d: Cost on batch %d is %f\n',i,cnn.epochs,batch_num,cost);
    end
    cnn.alpha = cnn.alpha/2.0;
    batch_num = 0;
end
toc
[~,~,preds]=cnn_train2(cnn,testImages,testLabels,true);
right = sum(preds==testLabels)/length(preds);
fprintf('Accuracy is %f\n',right);
