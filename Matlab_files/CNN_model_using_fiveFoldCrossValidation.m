%% Training a classifier on the CUB_200_2011 dataset using CNN
% with resized cropped images as input and cross-validation procedure
% Authors: Tien Dat Nguyen 
% Date created: 24/04/2023
% Date last updated: 30/04/2023

close all;
clear variables;
existing_GUIs = findall(0);
if length(existing_GUIs) > 1
    delete(existing_GUIs);
end
clc;

%% Read the image data from the relevant text files. 
folder = "C:/Users/melio/OneDrive - University of Canberra " + ...
    "(student)/UC - IT/Sem 1 - 2023/8890 - " + ...
    "Computer Vision and Image Analysis/CUB_200_2011/";
imgFolder = folder + "images/";
imgTxtFolder = folder + "images.txt";

% Load in all images from the dataset folder into one datastore
allImageDS = imageDatastore(imgFolder, 'IncludeSubfolders', true, ...
                            'LabelSource','foldernames');

%% Split dataset into five folds (=partitions) for fivefold cross-validation.
% Split dataset into 5 x 20% 
[fold1DS, fold2DS, fold3DS, fold4DS, fold5DS] = ...
    splitEachLabel(allImageDS, 0.2, 0.2, 0.2, 0.2);

% Set target size for common width and height after cropping
targetSize = [224, 224];

% Number of folds is five in this experiment
numFolds = 5;

%% Create a simple CNN
layers = [
    imageInputLayer([224 224 3])    % This needs to match the image size chosen above
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(200)
    softmaxLayer
    classificationLayer];

%% Check if we have a GPU available and clear any old data from it
if (gpuDeviceCount() > 0)
    device = gpuDevice(1);
    reset(device);  % Clear previous values that might still be on the GPU
end

%% Train the simple CNN model for each fold
accuracy_overall = 0.0;
for i = 1:numFolds
    [cdsTraining, cdsValidation, cdsTest, trainingImageDS, ...
        validationImageDS, testImageDS] = ...
        getFoldsFor5FoldCrossVal(i, fold1DS, fold2DS, fold3DS, fold4DS, ...
                                 fold5DS, folder, imgTxtFolder, targetSize);

    % Set the training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 20, ...
        'Verbose', true, ...
        'Shuffle', 'every-epoch', ...
        'VerboseFrequency', 1, ...
        'ValidationData', cdsValidation, ...
        'Plots','training-progress');

    simpleCNN = trainNetwork(cdsTraining, layers, options);

    YPred = classify(simpleCNN, cdsTest);
    YTest = testImageDS.Labels;
    
    accuracy = sum(YPred == YTest)/numel(YTest); % Output on command line
    disp("Accuracy for Run "+ string(i)+" is: " + accuracy);
    
    accuracy_overall = accuracy_overall+accuracy;
end

disp("Average accuracy of five folds is "+ string(accuracy_overall/numFolds))
