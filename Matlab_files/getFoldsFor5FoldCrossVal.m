function [cdsTraining, cdsValidation, cdsTest, trainingImageDS, ...
    validationImageDS, testImageDS] = ...
    getFoldsFor5FoldCrossVal(iRun, fold1DS, fold2DS, fold3DS, fold4DS, ...
    fold5DS, folder, imgTextFolder, targetSize)
    
    % As per the assignment description, we set up different trainig,
    % validation and test fold combinations by assigning the folds differently
    % for each of the five runs in the fivefold cross-validation.
    if iRun == 1
        % Run 1: Folds 1-3 for training, Fold 4 for validation, Fold 5 for test 
        trainingImageDS = imageDatastore(cat(1, fold1DS.Files, ...
                                             fold2DS.Files, fold3DS.Files));
        trainingImageDS.Labels = cat(1, fold1DS.Labels, fold2DS.Labels, ...
                                     fold3DS.Labels);
        validationImageDS = fold4DS;
        testImageDS = fold5DS;
    elseif iRun == 2
        % Run 2: Parts 2-4 for training, Part 5 for validation, Part 1 for test 
        trainingImageDS = imageDatastore(cat(1, fold2DS.Files, ...
                                             fold3DS.Files, fold4DS.Files));
        trainingImageDS.Labels = cat(1, fold2DS.Labels, fold3DS.Labels, ...
                                     fold4DS.Labels);
        validationImageDS = fold5DS;
        testImageDS = fold1DS;
    elseif iRun == 3
        % Run 3: Parts 3-5 for training, Part 1 for validation, Part 2 for test 
        trainingImageDS = imageDatastore(cat(1, fold3DS.Files, ...
                                             fold4DS.Files, fold5DS.Files));
        trainingImageDS.Labels = cat(1, fold3DS.Labels, fold4DS.Labels, ...
                                     fold5DS.Labels);
        validationImageDS = fold1DS;
        testImageDS = fold2DS;
    elseif iRun == 4  
        % Run 4: Parts 4, 5, and 1 for training, Part 2 for validation, Part 3 for test 
        trainingImageDS = imageDatastore(cat(1, fold4DS.Files, ...
                                             fold5DS.Files, fold1DS.Files));
        trainingImageDS.Labels = cat(1, fold4DS.Labels, fold5DS.Labels, ...
                                     fold1DS.Labels);
        validationImageDS = fold2DS;
        testImageDS = fold3DS;
    elseif iRun == 5
        %Run 5: Parts 5, 1, and 2 for training, Part 3 for validation, Part 4 for test
        trainingImageDS = imageDatastore(cat(1, fold5DS.Files, ...
                                             fold1DS.Files, fold2DS.Files));
        trainingImageDS.Labels = cat(1, fold5DS.Labels, fold1DS.Labels, ...
                                     fold2DS.Labels);
        validationImageDS = fold3DS;
        testImageDS = fold4DS;
    end
    
    % Get training, validation and testing image file names
    trainingImageNames = trainingImageDS.Files;
    validationImageNames = validationImageDS.Files;
    testImageNames = testImageDS.Files;
    
    % Read class info from the relevant text files - may not required
    classNames = readtable(folder + "classes.txt", ...
        'ReadVariableNames', false);
    classNames.Properties.VariableNames = {'index', 'className'};
    
    imageClassLabels = readtable(folder + "image_class_labels.txt", ...
        'ReadVariableNames', false);
    imageClassLabels.Properties.VariableNames = {'index', 'classLabel'};
    
    % Read bounding box information from bounding_boxes.txt. for cropping.
    % The format is: image index, x-coordinate top-left corner, 
    % y-coordinate top-left corner, width, height.
    boundingBoxes = readtable(folder + "bounding_boxes.txt", ... 
        'ReadVariableNames', false);
    boundingBoxes.Properties.VariableNames = {'index', 'x', 'y', 'w', 'h'};
    
    % Map bounding box information to the respective image file name
    train_image_box_map = returnMapping(trainingImageNames, boundingBoxes, ...
        imgTextFolder);
    val_image_box_map = returnMapping(validationImageNames, boundingBoxes, ...
        imgTextFolder);
    test_image_box_map = returnMapping(testImageNames, boundingBoxes, ...
        imgTextFolder);
    
    % Crop images to the bounding box area while reading in the image data
    trainingImageDS.ReadFcn = @(filename) ...
        readImagesIntoDatastoreBB_Fast(filename, train_image_box_map);
    validationImageDS.ReadFcn = @(filename) ...
        readImagesIntoDatastoreBB_Fast(filename, val_image_box_map);
    testImageDS.ReadFcn = @(filename) ...
        readImagesIntoDatastoreBB_Fast(filename, test_image_box_map);
    
    % Combine transformed datastores and labels 
    labelsTraining = arrayDatastore(trainingImageDS.Labels);
    labelsValidation = arrayDatastore(validationImageDS.Labels);
    labelsTest = arrayDatastore(testImageDS.Labels);
    
    cdsTraining = combine(trainingImageDS, labelsTraining);
    cdsValidation = combine(validationImageDS, labelsValidation);
    cdsTest = combine(testImageDS, labelsTest);
    
    % Resize all images to a common width and height
    cdsTraining = transform(cdsTraining, @(x) preprocessData(x, targetSize));
    cdsValidation = transform(cdsValidation, @(x) preprocessData(x, targetSize));
    cdsTest = transform(cdsTest, @(x) preprocessData(x, targetSize));
    
    %% Helper function for resizing images in transform
    function data_out = preprocessData(data, targetSize)
        try
            data_out{1} = imresize(data{1}, targetSize(1:2)); % Resize images
            data_out{2} = data{2};  % Keep labels as they are
        catch e
            % This is solely for debugging
            disp(e) 
        end
    end
    
    %% Helper function mapping image names to bounding boxes and vice versa
    function image_box_map = returnMapping(ImageNames, boundingBoxes, imageTxtPath)
        M = readlines(imageTxtPath);
        image_box_map = containers.Map;
        for i = 1:size(ImageNames, 1) 
            %fn = ImageNames{i};
            fn = split(ImageNames{i}, ["/", "\"]); % WinOS/MacOS/Linux
            ix = find(contains(M, fn{end}));
            image_box_map(fn{end}) = [boundingBoxes{ix, 2:5}];
        end
    end

end
