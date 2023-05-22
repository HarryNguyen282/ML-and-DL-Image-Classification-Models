%% Training a SVM classifier on the CUB_200_2011 dataset using SIFT and HOG
% features and bounding box image areas as input
%
% Author: Tien Dat Nguyen
% Date created: 19/04/2022
% Date last updated: 30/04/2023

close all;
clear variables;
existing_GUIs = findall(0);
if length(existing_GUIs) > 1
    delete(existing_GUIs);
end

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
disp('Training set class distribution:');
countEachLabel(trainingImageDS)

disp('Validation set class distribution:');
countEachLabel(validationImageDS)

disp('Test set class distribution:');
countEachLabel(testImageDS)

%% Apply data preprocessing to the training, validating and testing datastores

% Define target size for the training, validating and testing images
targetSize = [100, 100];
trainingImageDS_Resized = transform(trainingImageDS, @(x) imresize(x,targetSize));
validationImageDS_Resized = transform(validationImageDS, @(x) imresize(x,targetSize));
testImageDS_Resized = transform(testImageDS, @(x) imresize(x,targetSize));

% Combine transformed datastores and labels
labelsTraining = arrayDatastore(trainingImageDS.Labels);
cdsTraining = combine(trainingImageDS_Resized, labelsTraining);
labelsValidation = arrayDatastore(validationImageDS.Labels);
cdsValidation = combine(validationImageDS_Resized, labelsValidation);
labelsTest = arrayDatastore(testImageDS.Labels);
cdsTest = combine(testImageDS_Resized, labelsTest);

%% Display a sample image from the datastore
cellSize = [32 32];
figure(1);
img = cdsTraining.read{1};
[hog_32x32, vis_32x32] = extractHOGFeatures(rgb2gray(img), 'CellSize',cellSize);
SIFTpoints = detectSIFTFeatures(rgb2gray(img));
subplot(1, 3, 1);
imshow(img);
title('Sample Image Resized');
subplot(1, 3, 2);
imshow(rgb2gray(img)); hold on;
title('SIFT Feature Points');
plot(SIFTpoints.selectStrongest(20));
hold off;
subplot(1,3,3);
plot(vis_32x32);
title({'HOG CellSize = [32 32]'; ['Length = ' num2str(length(hog_32x32))]});


%% Extract HOG & SIFT features
% Define HOG feature size
hogFeatureSize = length(hog_32x32);

% Define limits for SIFT feature size
numFeatures = 5;
maxFeatures = 20;

% Start by extracting features from the training, validating, and test datastores via the helper function.
% These features will be used to train the classifier. 
[trainingFeatures, trainingLabels] = ...
    helperExtractHOGandSIFTFeaturesFromImageSet(cdsTraining, numFeatures, maxFeatures, hogFeatureSize, cellSize);

% These features will be used to test the classifier
[testFeatures, testLabels] = ...
    helperExtractHOGandSIFTFeaturesFromImageSet(cdsTest,  numFeatures, maxFeatures, hogFeatureSize, cellSize);
%% Check if we have a GPU available and clear any old data from it
if (gpuDeviceCount() > 0)
    disp('Found GPU:');
    disp(gpuDeviceTable);
    device = gpuDevice(1);
    reset(device);  % Clear previous values that might still be on the GPU
end

%% Train a multi-class SVM
t = templateSVM('KernelFunction', 'gaussian');
options = struct('UseParallel', true);

% fitcecoc uses multiple SVM learners and a 'One-vs-One' encoding scheme.
SVMClassifier = fitcecoc(trainingFeatures, trainingImageDS.Labels,'Learners', t, ...
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', options);%

%% Test the accuracy on the test partition
YPred = predict(SVMClassifier, testFeatures);
YTest = testImageDS.Labels;

% Calculate overall accuracy
accuracy = sum(YPred == YTest)/numel(YTest) % Output on command line

% Create confusion matrix for data analytics
[m, order] = confusionmat(YTest, YPred);

%% Compute classwise positive recogniton rate
classwisePosRecog = zeros(height(order), 1);
samplesPerRow = sum(m, 2);
% Display the class-wise recognition rate to command line
disp('Classwise Recognition Rates:');
for iI = 1:height(order)
    rate = round(100 * m(iI, iI) / samplesPerRow(iI), 1);
   classwisePosRecog(iI,2) = rate;
   output = num2str(iI) + ":" + " " + num2str(rate) + "%";
   disp(output);
end
