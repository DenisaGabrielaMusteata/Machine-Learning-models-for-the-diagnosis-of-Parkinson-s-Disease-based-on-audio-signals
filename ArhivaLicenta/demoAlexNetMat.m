
clear all
close all
clc

%% Parameters

% DataSet
pathImages="C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\SpectrogrameMel";
        %imagini RGB 227 x 227

% Results file
nameFileRez='rezF4096_NB64';

% train parameters
MBS=50; %mini-batch  
NoEp=100; %nr of epochs


%% Database creation with RGB images for train, test and validation
imds = imageDatastore(pathImages, 'FileExtension', '.mat',...  
    'IncludeSubfolders',true, 'ReadFcn', @sampleMatReader, ...
    'LabelSource','foldernames'); 
 
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds,0.8,0.10,0.10,'randomized'); 

numTrainImages = numel(imdsTrain.Files); 
numClasses = numel(categories(imdsTrain.Labels));

im=readimage(imdsTrain,1); inputSize = size(im); 

%% Load the pre-trained model alexnet 
net = alexnet; 

%% Construction of the model architecture with "transfer learning": layers

% imported layers
layersTransfer = net.Layers(1:end-3); 

% new model
layers = [
    layersTransfer   
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) % ponderi initializate cu valori aletoare
    softmaxLayer 
    classificationLayer]; 

%% Train the network

% set the parameters for the training
options = trainingOptions('sgdm', ...
    'MiniBatchSize',NoEp,...            
    'MaxEpochs',MBS, ...      
    'InitialLearnRate',1e-4, ...  
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
                  
% CNN train
netTransfer = trainNetwork(imdsTrain,layers,options);
nameFileRez = [nameFileRez, '_', num2str(MBS), '_', num2str(NoEp), '.mat'];
% saving the model
feval(@save,nameFileRez,'netTransfer'); 


%% Verify the result after training - for the dataset for train/validation/test 
[YPredValidation,scoresValidation] = classify(netTransfer,imdsValidation);  % network response
accuracyValidation = mean(YPredValidation == imdsValidation.Labels)  % accuracy
 
[YPredTrain,scoresTrain] = classify(netTransfer,imdsTrain); 
accuracyTrain = mean(YPredTrain == imdsTrain.Labels)  

[YPredTest,scoresTest] = classify(netTransfer,imdsTest);  
accuracyTest = mean(YPredTest == imdsTest.Labels)  
 
