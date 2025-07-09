clear all;
clc;

%% Variables
aux_data = {}; % main cell array(contains id, signals, framea, classes etc)
audioFrames = {}; %cell array for the frames
row = 1;
k = 1 ;
frameDuration = 10;  % frame durations (10 seconds)
overlapDuration = 1; %  overlap durations (1 seconds)
Frequency = 44100; % frequency
L = Frequency * 10;
T = 1/Frequency;
t = (0:L-1) * T;
K = 2.15*10^4; % frequency range
featMat = []; % future matrix
classV = []; % class array
infoAux = []; % additional information array

%% DataSet acces
listRead = dir("C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\AllSamples_Read");
audioDirectory_Read = "C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\AllSamples_Read";
audioFiles_Read = dir(fullfile(audioDirectory_Read, '*.wav'));
audioDirectory_Dialog = "C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\AllSamples_Dialog";
audioFiles_Dialog = dir(fullfile(audioDirectory_Dialog, '*.wav'));
cell = {'Read_sample';'Dialog_Sample'};


%% Id and class extraction
for i = 3:length(listRead)
    namefile = listRead(i).name; % read the id signal
    subject_class = namefile(6:7); % read the class
    aux_data{row,1} = namefile; % add the id in the cell array
    %classification
    if subject_class == 'hc'
        aux_data{row,3} = 0;
    else subject_class == 'pd'
        aux_data{row,3} = 1;
    end
    row = row+1;
end

%% extragerea semnalelor
for i=1:length(aux_data)
    filename_Read = fullfile(audioDirectory_Read, audioFiles_Read(i).name);
    wav_file_Read = audioread(filename_Read);
    cell{1,2} = wav_file_Read;
    filename_Dialog = fullfile(audioDirectory_Dialog, audioFiles_Dialog(i).name);
    wav_file_Dialog = audioread(filename_Dialog);
    cell{2,2} = wav_file_Dialog;
    aux_data{i,2}=cell;
end

%% Framing

% frame and overlap duration
frameLength = round(frameDuration * Frequency);
overlapDurationLength = round(overlapDuration * Frequency);

%% Framin the read samples
for i=1:length(aux_data)
    audioSample_Reading = aux_data{i,2}{1,2};
    startSample = 1;
    endSample = frameLength;
    nr = 0;
    while endSample <= length(audioSample_Reading)
        frame = audioSample_Reading(startSample:endSample);
        nr = nr+1;
        audioFrames{k,1} = ['S_',num2str(i),'_frame_',num2str(nr)];
        audioFrames{k,2} = frame;
        audioFrames{k,3} = aux_data{i,3};
        startSample = startSample + (frameLength - overlapDurationLength);
        endSample = startSample + frameLength - 1;
        k = k+1;
        aux_data{i,2}{1,3}=audioFrames;
    end
    audioFrames = {};
    k = 1;
end

%% Framing the spontanous dialog sample
for i=1:length(aux_data)
    audioSample_Dialog = aux_data{i,2}{2,2};
    startSample = 1; % starting the aux_data table
    endSample = frameLength;
    nr = 0;
    while endSample <= length(audioSample_Dialog)
        frame = audioSample_Dialog(startSample:endSample);
        nr = nr+1;
        audioFrames{k,1} = ['S_',num2str(i),'_frame_',num2str(nr)];
        audioFrames{k,2} = frame;
        audioFrames{k,3} = aux_data{i,3};
        startSample = startSample + (frameLength - overlapDurationLength);
        endSample = startSample + frameLength - 1;
        k = k+1;
        aux_data{i,2}{2,3}=audioFrames;
    end
    audioFrames = {};
    k = 1;
end

%% Future matrix and classes vector
for i = 1:length(aux_data)

    % Future matrix
    for j = 1:length(aux_data{i,2}{1,3})
        frame_fft_R_C = fft(aux_data{i,2}{1,3}{j,2});
        frame_fft_R = abs(frame_fft_R_C/L);
        featMat = [featMat;frame_fft_R(1:K)'];
    end
    for j = 1:length(aux_data{i,2}{2,3})
        frame_fft_D_C = fft(aux_data{i,2}{2,3}{j,2});
        frame_fft_D = abs(frame_fft_D_C/L);
        featMat = [featMat;frame_fft_D(1:K)'];
    end

    % Classes vector
    for j = 1:length(aux_data{i,2}{1,3})
        classV = [classV; aux_data{i,2}{1,3}{j,3}];
    end
    for j = 1:length(aux_data{i,2}{2,3})
        classV = [classV; aux_data{i,2}{2,3}{j,3}];
    end

    % Id vector
    for j = 1:length(aux_data{i,2}{1,3})
        a{1}= aux_data{i,2}{1,3}{j,1};
        infoAux = [infoAux;a];
    end
    for j = 1:length(aux_data{i,2}{2,3})
        a{1}= aux_data{i,2}{2,3}{j,1};
        infoAux = [infoAux;a];
    end
end

%% Partitioning
test = 20;
train = 80;
cv = cvpartition(size(featMat,1), 'Holdout', train/100);
indexTrain = cv.training;
indexTest = cv.test;

%% DataSets for training and testing
dataTrain = featMat(indexTrain,:);
dataTest = featMat(indexTest,:);

trainingClass = classV(indexTrain);
testClass = classV(indexTest);

%% Dimensions
disp(['Training set: ' num2str(size(dataTrain))]);
disp(['Test set: ' num2str(size(dataTest))]);


%% Random Forest training

nrTrees = 50;
modelC = fitensemble(dataTrain, trainingClass, 'Bag', nrTrees, 'Tree', 'type', 'classification');
[y_model_train,scoreTrain] = predict(modelC, dataTrain);
y_model_test = predict(modelC, dataTest);

accuracy_train = sum(y_model_train == trainingClass) / numel(trainingClass) * 100;
accuracy_test = sum(y_model_test == testClass) / numel(testClass) * 100;

fprintf('Accuracy percentage training set: %.2f%%\n', accuracy_train);
fprintf('Accuracy percentage testing set: %.2f%%\n', accuracy_test);

%% Confusion Matrix
confusionMat_test = confusionmat(testClass, y_model_test);
confusionMat_train = confusionmat(trainingClass, y_model_train);

%% Mel-spectrograms with Phen
for i = 1 : size(featMat, 1)
    frame = featMat(i, :)';
    calculateSpectrogram(frame, 2^Phen(1,1), 2^Phen(1,2), Frequency);
end
%% Mel-spectrograms generation 
function spectrograma = calculateSpectrogram(frame, window, nb, frequency)
fisierComun = "C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\SpectrogrameMel";
fisierSanatosiMel = "C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\SpectrogrameMel\0";
fisierBolnaviMel = "C:\Users\muste\Desktop\licenta\26-29_09_2017_KCL\SpectrogrameMel\1";
% interation for each row in the mat and apply the melSpectrogram function
dim = [227,227];
% calculate the mel-spectrogram for each frame
S = melSpectrogram(frame, Frequency, ...
    'Window', hann(window,'periodic'), ...
    'OverlapLength', window/4, ...
    'FFTLength', 2*window, ... 
    'NumBands', nb, ...
    'FrequencyRange', [0, Frequency/2]); 

% spectrogram redimension
spectrogramResized = imresize(S*10^14, dim);

S_new = abs(10*log10(S+eps));
spectrogramResized = imresize(S_new, dim);

rgbSpectrogram(:,:,1) = (spectrogramResized);
rgbSpectrogram(:,:,2) = (spectrogramResized);
rgbSpectrogram(:,:,3) = (spectrogramResized);

% save the redimensioned spectrogram depending of the class
if classV(i) == 0
    filename = fullfile(fisierSanatosiMel, sprintf('%d.mat', i));
    save(filename, 'rgbSpectrogram');
elseif classV(i) == 1
    filename = fullfile(fisierBolnaviMel, sprintf('%d.mat', i));
    save(filename, 'rgbSpectrogram');
end

end

