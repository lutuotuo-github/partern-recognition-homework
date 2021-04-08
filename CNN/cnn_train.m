function [cost, grad, preds] = cnn_train(theta,images,labels,numClasses,filterDim,numFilters,poolDim,activationType,pred)

imageDim = size(images,1); 
numImages = size(images,3);
outDim = (imageDim - filterDim + 1)/poolDim;
hiddennet = outDim^2*numFilters;

m = 1;
n = filterDim^2*numFilters;
w_filters = reshape(theta(m:n),filterDim,filterDim,numFilters);
m = n+1;
n = n+hiddennet*numClasses;
w_net = reshape(theta(m:n),numClasses,hiddennet);

convDim = imageDim-filterDim+1; 
outputDim = (convDim)/poolDim; 
activations = zeros(convDim,convDim,numFilters,numImages);
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
%% Ç°Ïò
meanPooling = ones(poolDim, poolDim);
Wc_rotated = zeros(size(w_filters));

for filterNum = 1 : numFilters
    Wc_rotated(:, :, filterNum) = rot90(w_filters(:, :, filterNum),2);
end
areaOfPoolingFilter = poolDim ^ 2;
meanPooling = meanPooling / areaOfPoolingFilter;
poolingIndex = 1 : poolDim : size(conv2(conv2(images(:, :, 1), Wc_rotated(:, :, 1), 'valid'), meanPooling, 'valid'), 1);
for imageNum = 1 : numImages
    image = images(:, :, imageNum);
    for filterNum = 1 : numFilters
        filteredImage = conv2(image, Wc_rotated(:, :, filterNum), 'valid');
        switch activationType
            case 'relu'
                filteredImage = max(filteredImage, 0);
            case 'sigmoid'
                filteredImage = sigmoid(filteredImage);
        end
        activations(:, :, filterNum, imageNum) = filteredImage;
        pooledImage = conv2(filteredImage, meanPooling, 'valid');
        activationsPooled(:, :, filterNum, imageNum) = pooledImage(poolingIndex, poolingIndex);
    end
end

%% Îó²î
activationsPooledReshaped = reshape(activationsPooled,[],numImages);
p_out = zeros(numClasses,numImages);
out_Softmax = w_net * activationsPooledReshaped ;
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

errorsSoftmax = p_out - onehotLabels;
errorsSoftmax = errorsSoftmax / numImages;
errorsPooled = w_net' * errorsSoftmax;
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);
errorsPooling = zeros(convDim, convDim, numFilters, numImages);
for imageNum = 1:numImages
    % for imageNum = 1:numImages
    for filterNum = 1:numFilters
        e = errorsPooled(:, :, filterNum, imageNum);
        errorsPooling(:, :, filterNum, imageNum) = kron(e, meanPooling);
    end
end
switch activationType
    case 'relu'
        errorsConvolution = errorsPooling .* (activations > 0); % relu derivative = x > 1
    case 'sigmoid'
        errorsConvolution = errorsPooling .* activations .* (1 - activations); % sigmoid derivative = x .* (1 - x)
end

%% ÌÝ¶È
wnet_grad = errorsSoftmax * activationsPooledReshaped';
wfilter_grad = zeros(size(w_filters));
for filterNum = 1 : numFilters
    for imageNum = 1 : numImages
        e = errorsConvolution(:, :, filterNum, imageNum);
        errorsConvolution(:, :, filterNum, imageNum) = rot90(e,2);
    end
end

for filterNum = 1 : numFilters
    Wc_gradFilter = zeros(size(wfilter_grad, 1), size(wfilter_grad, 2));
    for imageNum = 1 : numImages
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), errorsConvolution(:, :, filterNum, imageNum), 'valid');
    end
    wfilter_grad(:, :, filterNum) = Wc_gradFilter;
end

grad = [wfilter_grad(:) ; wnet_grad(:)];
end
