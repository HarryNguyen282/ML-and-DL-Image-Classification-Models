%% Training a classifier on the CUB_200_2011 dataset using CNN
% with augmenting cropped images as input
% Authors: Tien Dat Nguyen 
% Date created: 18/04/2023
% Date last updated: 30/04/2023

close all;
clear variables;
existing_GUIs = findall(0);
if length(existing_GUIs) > 1
    delete(existing_GUIs);
end
clc;

%% Read the training, validation and test partitions from the relevant
%  text files. 

% Define the folder containing dataset
folder = "C:/Users/melio/OneDrive - University of Canberra " + ...
    "(student)/UC - IT/Sem 1 - 2023/8890 - " + ...
    "Computer Vision and Image Analysis/CUB_200_2011/";
trainingImageNames = readtable(fullfile(folder, "train.txt"), ... 
    'ReadVariableNames', false);
trainingImageNames.Properties.VariableNames = {'index', 'imageName'};

validationImageNames = readtable(folder + "validate.txt", ... 
    'ReadVariableNames', false);
validationImageNames.Properties.VariableNames = {'index', 'imageName'};

testImageNames = readtable(folder + "test.txt", ... 
    'ReadVariableNames', false);
testImageNames.Properties.VariableNames = {'index', 'imageName'};

%% Read class info from the relevant text files
classNames = readtable(folder + "classes.txt", ...
    'ReadVariableNames', false);
classNames.Properties.VariableNames = {'index', 'className'};

imageClassLabels = readtable(folder + "image_class_labels.txt", ...
    'ReadVariableNames', false);
imageClassLabels.Properties.VariableNames = {'index', 'classLabel'};

%% Read bounding box information from bounding_boxes.txt. The format is
%  image index, x-coordinate top-left corner, y-coordinate top-left corner,
%  width, height.
boundingBoxes = readtable(folder + "bounding_boxes.txt", ... 
    'ReadVariableNames', false);
boundingBoxes.Properties.VariableNames = {'index', 'x', 'y', 'w', 'h'};

% Map bounding box information to the respective image file name
train_image_box_map = returnMapping(trainingImageNames, boundingBoxes);
val_image_box_map = returnMapping(validationImageNames, boundingBoxes);
test_image_box_map = returnMapping(testImageNames, boundingBoxes);

%% Create lists of image names for training, validation and test subsets.
%  The data stored in the variables are the full file path to the images
trainingImageList = strings(height(trainingImageNames), 1);
for iI = 1:height(trainingImageNames)
    trainingImageList(iI) = string(fullfile(folder, "images/", ...
        string(cell2mat(trainingImageNames.imageName(iI)))));
end

validationImageList = strings(height(validationImageNames), 1);
for iI = 1:height(validationImageNames)
    validationImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(validationImageNames.imageName(iI))));
end

testImageList = strings(height(testImageNames), 1);
for iI = 1:height(testImageNames)
    testImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(testImageNames.imageName(iI))));
end

%% Create image datastores for training, validation and test subsets
% and make sure images are cropped to the area inside the appropriate
% pre-defined bouding box as well as having 3 colour channels
trainingImageDS = imageDatastore(trainingImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
trainingImageDS.ReadFcn = @(filename) readImagesIntoDatastoreBB_Fast(filename, train_image_box_map);

validationImageDS = imageDatastore(validationImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
validationImageDS.ReadFcn = @(filename) readImagesIntoDatastoreBB_Fast(filename, val_image_box_map);

testImageDS = imageDatastore(testImageList, 'labelSource', 'foldernames',   'FileExtensions', {'.jpg'});
testImageDS.ReadFcn = @(filename) readImagesIntoDatastoreBB_Fast(filename, test_image_box_map);

%% Display the image class distribution
disp('Training set class samples:');
countEachLabel(trainingImageDS)
disp('Validation set class samples:');
countEachLabel(validationImageDS)
disp('Test set class samples:');
countEachLabel(testImageDS)

%% Apply data preprocessing to the training, validating and testing datastores

% Define target size for the training, validating and testing images
targetSize = [224 224];

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXReflection',true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation',[-3 3]);

% Create augmented image datastores for training, validating, and testing
% partition
trainingAugImds = augmentedImageDatastore(targetSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationAugImds = augmentedImageDatastore(targetSize, validationImageDS); % resize only
testingAugImds = augmentedImageDatastore(targetSize,testImageDS); % resize only

%% Display a sample image from the datastores

figure(1);
subplot(1,3,1);
imshow(read(trainingAugImds).input{1});
title('Training Image');
subplot(1,3,2);
imshow(read(validationAugImds).input{1});
title('Validating Image');
subplot(1,3,3);
imshow(read(testingAugImds).input{1});
title('Testing Image');
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

%% Set the training options
options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 20, ...
        'Verbose', true, ...
        'Shuffle', 'every-epoch', ...
        'VerboseFrequency', 1, ...
        'ValidationData', validationAugImds, ...
        'Plots','training-progress');

%% Train the simple CNN model

% Make sure to start from the beginning of each set
reset(trainingAugImds);
reset(validationAugImds);
reset(testingAugImds);

simpleCNN = trainNetwork(trainingAugImds, layers, options);

%% Test the accuracy on the test partition
YPred = classify(simpleCNN, testingAugImds);
YTest = testImageDS.Labels;

% Calculate overall accuracy
accuracy = sum(YPred == YTest)/numel(YTest) % Output on command line

% Create confusion matrix for data analytics
[m, order] = confusionmat(YTest, YPred);

%% Compute classwise positive recogniton rate
classwisePosRecog = zeros(height(order), 1);
samplesPerRow = sum(m, 2);
% Display to command line
disp('Classwise Recognition Rates:');
for iI = 1:height(order)
    rate = round(100 * m(iI, iI) / samplesPerRow(iI), 1);
   classwisePosRecog(iI,2) = rate;
   output = num2str(iI) + ":" + " " + num2str(rate) + "%";
   disp(output);
end