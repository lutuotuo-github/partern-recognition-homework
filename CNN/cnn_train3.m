function [cost, grad, preds] = cnn_train3(cnn,minibatch_data,minibatch_labels,pred)

theta = cnn.theta;
images = minibatch_data;
labels = minibatch_labels;
numClasses = cnn.numClasses;
filterDim1 = cnn.layers{2, 1}.kernelsize;
filterDim2 = cnn.layers{4, 1}.kernelsize;
numFilters1 = cnn.layers{2, 1}.outputmaps;
numFilters2 = cnn.layers{4, 1}.outputmaps;
poolDim1 = cnn.layers{3, 1}.scale;
poolDim2 = cnn.layers{5, 1}.scale;
activationType = cnn.activationType;

imageDim = size(images,1); 
numImages = size(images,3);
outDim1 = (imageDim - filterDim1 + 1)/poolDim1;
outDim2 = (outDim1 - filterDim2 + 1)/poolDim2;
hiddennet = outDim2^2*numFilters2;

m = 1;
n = filterDim1^2*numFilters1;
w_filters1 = reshape(theta(m:n),filterDim1,filterDim1,numFilters1);
m = n+1;
n = n+filterDim2^2*numFilters2*numFilters1;
w_filters2 = reshape(theta(m:n),filterDim2,filterDim2,numFilters2,numFilters1);
m = n+1;
n = n+hiddennet*numClasses;
w_net = reshape(theta(m:n),numClasses,hiddennet);

convDim = outDim1-filterDim2+1; 
outputDim = (convDim)/poolDim2; 

activations = zeros(convDim,convDim,numFilters2,numImages);
activationsPooled = zeros(outputDim,outputDim,numFilters2,numImages);
%% 前向传播
meanPooling = ones(poolDim2, poolDim2);
Wc_rotated2 = zeros(size(w_filters2));
Wc_rotated1 = zeros(size(w_filters1));
for filterNum = 1 : numFilters2
    for j = 1 : numFilters1
    Wc_rotated2(:, :, filterNum,j) = rot90(w_filters2(:, :,  filterNum,j),2);
    end
end
for filterNum = 1 : numFilters1
    Wc_rotated1(:, :,filterNum) = rot90(w_filters1(:, :, filterNum),2);
end

areaOfPoolingFilter = poolDim2 ^ 2;
meanPooling = meanPooling / areaOfPoolingFilter;
poolingIndex1 = 1 : poolDim2 : size(conv2(conv2(images(:, :, 1), Wc_rotated1(:, :, 1), 'valid'), meanPooling, 'valid'), 1);
poolingIndex2 = 1 : poolDim2 : size(conv2(conv2(images(:, :, 1), Wc_rotated2(:, :, 1), 'valid'), meanPooling, 'valid'), 1);
for imageNum = 1 : numImages
    image = images(:, :, imageNum);
    for filterNum = 1 : numFilters1
        filteredImage = conv2(image, Wc_rotated1(:, :, filterNum), 'valid');
        switch activationType
            case 'relu'
                filteredImage = max(filteredImage, 0);
            case 'sigmoid'
                filteredImage = sigmoid(filteredImage);
        end
        activations1(:, :, filterNum, imageNum) = filteredImage;
        pooledImage1 = conv2(filteredImage, meanPooling, 'valid');
        activationsPooled1(:, :, filterNum, imageNum) = pooledImage1(poolingIndex1, poolingIndex1);
    end
end
poolingIndex2 = 1 : poolDim2 : size(conv2(conv2(activationsPooled1(:, :, 1,1), Wc_rotated2(:, :, 1,1), 'valid'), meanPooling, 'valid'), 1);
for imageNum = 1 : numImages
%     activationsPooled1 = images(:, :, imageNum);
    for j = 1 : numFilters2  %6
%         
        filteredImage = 0;
        for outNum1 = 1: numFilters1  %12
            outimages1 = activationsPooled1(:, :, outNum1,imageNum);
            filteredImage = filteredImage + conv2(outimages1, Wc_rotated2(:, :,j, outNum1), 'valid');
        end
        switch activationType
            case 'relu'
                filteredImage = max(filteredImage, 0);
            case 'sigmoid'
                filteredImage = sigmoid(filteredImage);
        end
        activations2(:, :, j, imageNum) = filteredImage;
        pooledImage2 = conv2(filteredImage, meanPooling, 'valid');
        activationsPooled2(:, :, j, imageNum) = pooledImage2(poolingIndex2, poolingIndex2);
    end
end


activationsPooledReshaped2 = reshape(activationsPooled2,[],numImages);
p_out = zeros(numClasses,numImages);
out_Softmax = w_net * activationsPooledReshaped2 ;
out_Softmax = exp(out_Softmax-max(out_Softmax));
p_out = out_Softmax./sum(out_Softmax);

cost = 0; 
labelIndex = sub2ind(size(out_Softmax), labels', 1:numImages);
onehotLabels = zeros(size(out_Softmax));
onehotLabels(labelIndex) = 1;
cost = -sum(sum(onehotLabels .* log(p_out)))/numImages;

if pred
    [~,preds] = max(p_out,[],1);
    preds = preds';
    grad = 0;
    return;
end
%% 反向传播并计算梯度
errorsSoftmax = p_out - onehotLabels;
errorsSoftmax = errorsSoftmax / numImages;
errorsPooled2 = w_net' * errorsSoftmax;

errorsPooled2 = reshape(errorsPooled2, [], outputDim, numFilters2, numImages);
errorsPooling2 = zeros(convDim, convDim, numFilters2, numImages);
for imageNum = 1:numImages
    % for imageNum = 1:numImages
    for filterNum = 1:numFilters2
        e = errorsPooled2(:, :, filterNum, imageNum);
        errorsPooling2(:, :, filterNum, imageNum) = kron(e, meanPooling);
    end
end

switch activationType
    case 'relu'
        errorsConvolution2 = errorsPooling2 .* (activations2 > 0); % relu derivative = x > 1
    case 'sigmoid'
        errorsConvolution2 = errorsPooling2 .* activations2 .* (1 - activations2); % sigmoid derivative = x .* (1 - x)
end


% 计算梯度
wnet_grad = errorsSoftmax * activationsPooledReshaped2';

% 两层卷积的误差



wfilter2_grad = zeros(size(w_filters2));
for filterNum = 1 : numFilters2
    for imageNum = 1 : numImages
        e = errorsConvolution2(:, :, filterNum, imageNum);
        errorsConvolution2(:, :, filterNum, imageNum) = rot90(e,2);
    end
end
for i = 1 : numFilters1
    Wc_gradFilter2 = zeros(size(wfilter2_grad, 1), size(wfilter2_grad, 2));
    for imageNum = 1 : numImages
        for j = 1 : numFilters2
            Wc_gradFilter2 = Wc_gradFilter2 + conv2(activationsPooled1(:, :, i, imageNum), ...
                errorsConvolution2(:, :, i, imageNum), 'valid');
        end
    end
    wfilter2_grad(:, :, i) = Wc_gradFilter2;
end

%第一层池化层后的误差

for imageNum = 1 : numImages
    for j = 1 : numFilters1  %6      
        filteredImage = 0;
        for outNum1 = 1: numFilters2  %12
            outimages1 = errorsConvolution2(:, :, outNum1,imageNum);
            filteredImage = filteredImage + conv2(outimages1, Wc_rotated2(:, :,outNum1,j), 'full');
        end
        errorsPooled1(:, :, j, imageNum) = filteredImage;
    end
end


% errorsPooled1 = errorsConvolution1;
% errorsPooling1 = zeros(convDim, convDim, numFilters2, numImages);
for imageNum = 1:numImages
    % for imageNum = 1:numImages
    for filterNum = 1:numFilters1
        e = errorsPooled1(:, :, filterNum, imageNum);
        errorsPooling1(:, :, filterNum, imageNum) = kron(e, meanPooling);
    end
end

switch activationType
    case 'relu'
        errorsConvolution1 = errorsPooling1 .* (activations1 > 0); % relu derivative = x > 1
    case 'sigmoid'
        errorsConvolution1 = errorsPooling1 .* activations1 .* (1 - activations1); % sigmoid derivative = x .* (1 - x)
end

wfilter1_grad = zeros(size(w_filters1));
for filterNum = 1 : numFilters1
    Wc_gradFilter1 = zeros(size(wfilter1_grad, 1), size(wfilter1_grad, 2));
    for imageNum = 1 : numImages
        Wc_gradFilter1 = Wc_gradFilter1 + conv2(images(:, :, imageNum), errorsConvolution1(:, :, filterNum, imageNum), 'valid');
    end
    wfilter1_grad(:, :, filterNum) = Wc_gradFilter1;
end

grad = [ wfilter1_grad(:);wfilter2_grad(:) ; wnet_grad(:)];
end
