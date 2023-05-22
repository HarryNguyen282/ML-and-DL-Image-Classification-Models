
function [outputFeatures, setLabels] = ...
helperExtractHOGandSIFTFeaturesFromImageSet(imds,  num_of_features, maxFeatureLimit, HogFeatureSize, cellsize)
    % Extract HOG and SIFT features from an imageDatastore
    
    setLabels = imds.UnderlyingDatastores{1,1}.UnderlyingDatastores{1,1}.Labels;
    numImages = numel(imds.UnderlyingDatastores{1,1}.UnderlyingDatastores{1,1}.Files);
    outputSIFTFeatures  = zeros(numImages,num_of_features * maxFeatureLimit,'single');
    outputHOGFeatures = zeros(numImages, HogFeatureSize, 'single');

    % Ensuring starting from the first image in the datastore
    reset(imds);

    % Process each image and extract features
    for j = 1:numImages
        imgFromDS = imds.read{1};
        img = imbinarize(im2gray(imgFromDS));
        
        % Detect SIFT Features
        points = detectSIFTFeatures(img);  
        [features, valid_points] = extractFeatures(img, points);
        points = valid_points.selectStrongest(maxFeatureLimit);

        if ~isempty(points) 
            for p = 0:length(points)-1  
                l = points(p+1).Location; 
                outputSIFTFeatures(j, (p*num_of_features)+1) = l(1);
                outputSIFTFeatures(j, (p*num_of_features)+2) = l(2);
                outputSIFTFeatures(j, (p*num_of_features)+3) = points(p+1).Scale;  
                outputSIFTFeatures(j, (p*num_of_features)+4) = points(p+1).Octave;
                %outputSIFTFeatures(j, (p*num_of_features)+5) = points(p+1).Orientation;
                outputSIFTFeatures(j, (p*num_of_features)+5) = points(p+1).Metric;
            end 
        end   
        % Detect HOG Features
        outputHOGFeatures(j, :) = extractHOGFeatures(img, 'CellSize', cellsize);
    end

    %outputFeatures = outputSIFTFeatures;
    % Combine both features to the output feature of the input image
    outputFeatures = cat(2, outputSIFTFeatures, outputHOGFeatures);
end % end of function