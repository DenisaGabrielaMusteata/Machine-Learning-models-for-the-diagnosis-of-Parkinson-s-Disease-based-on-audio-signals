
clear all
close all
clc

%% Parameters

nameFileRez='rezMyLayer1_thirdRun';

%Data
Frequency=44100;
p=0.7;

% NN
MBS=50; %mini-batch 
NoEpInit=100; %nr of epochs
withFig=0;

%GA
NIND = 8; % dimension of a population
MAXGEN = 10; % maximum number of generations
GGAP = 0.5; % select GGAP*NIND parents for reproduction
PC=0.7; % the probability of applying the crossover operator 
PM=0.1; % the probability of applying the mutation operator 

% matrix for a 4 individuals population
% FieldD=[2 2;8 4; 11 7; 0 0;0 0; 1 1; 1 1]; % codification matrix
% LIND =sum(FieldD(1,:)); % chromosome length

% matrix for a 8 individuals population
FieldD=[4 4;6 2; 13 9; 0 0;0 0; 1 1; 1 1]; % codification matrix
LIND =sum(FieldD(1,:)); % chromosome length


%% Database creation with RGB images for train, test and validation
load dataTrain.mat
load dataTest.mat
load classVtest.mat
load classVtrain.mat

dsIn=arrayDatastore(dataTrain(:,:));
dsL=arrayDatastore(categorical(classVtrain(:,:)));
dsTrain=combine(dsIn,dsL);
dsInVal=[];classVtrainVal=[];


labels=unique(categorical(classVtrain));

numTrainImages = size(dataTrain,1);
numClasses = numel(unique(classVtrain));
inputSize=size(dataTrain(1,:));

%% Load the pre-trained model alexnet
net = alexnet;

%% Construction of the model architecture with "transfer learning": layers

% imported layers
layersTransfer = net.Layers(3:end-3);
layerC1Old = net.Layers(2); % save the second layer in a new variable
layerC1new=convolution2dLayer(layerC1Old.FilterSize,layerC1Old.NumFilters,Name=layerC1Old.Name,Stride=layerC1Old.Stride,Bias=layerC1Old.Bias,Weights=layerC1Old.Weights(:,:,3,:));

% new model
layers = [
    imageInputLayer([227 227 1],'Name','data')
    layerC1new
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) % ponderi initializate cu valori aletoare
    softmaxLayer
    ];

netMy=dlnetwork(layers); %create the dinamic network

%% GA
numefct='evalChrom';
gen=0; % generation number - initialisation
Chrom = crtbp(NIND, LIND); % initial population (binary)
allObjV = [];

fprintf(1,"generatia %g ===================\n",gen);
Phen=fix(bs2rv(Chrom,FieldD)); % genoptip->fenotip (decodification)
[ObjV, allObjV] = feval(numefct,Phen,netMy, dsTrain, NoEpInit,MBS,classVtrain,Frequency,dsInVal,classVtrainVal,allObjV); % evalueaza populatia

%show the results (min function values)
Best(gen+1) = min(ObjV);
plot((Best),'ro');xlabel('generatie'); ylabel('functia obiectiv');
title(['Best = ', num2str(Best(gen+1))]);
drawnow;
for gen=1:1:MAXGEN
    fprintf(1,"generatia %g ===================\n",gen);
    FitnV = ranking(ObjV); % fitness values - rank mode

    SelCh = select('sus', Chrom, FitnV, GGAP); %select GGAP x NIND parents for reproduction;

    SelCh = recombin('recdis',SelCh,PC);  % crossover operator 
    SelCh = mut (SelCh,PM);   % mutation operator

    PhenSel=fix(bs2rv(SelCh,FieldD));
    [ObjVSel,allObjV] = feval(numefct,PhenSel,netMy, dsTrain, NoEpInit,MBS,classVtrain,Frequency,dsInVal,classVtrainVal,allObjV); %evaluare copii

    [Chrom ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); % inseration of the new soltuions in the matrix 

    Best(gen+1) = min(ObjV); % best ObjV value

    plot((Best),'ro');xlabel('generatie'); ylabel('functia obiectiv');
    title(['Best = ', num2str(Best(gen+1))]);
    drawnow;

end


% best solution
[v,idx]=min(ObjV);
Phen=fix(bs2rv(Chrom(idx(1),:),FieldD)); %valorile pt parametrii care variaza

%% Objective Function

function [ObjV,allObjV] = evalChrom(Phen,netMy, dsTrain, NoEP,MBS,classVtrain,Frequency,dsInVal,classVtrainVal,allObjV)

[Nchrom,Ngenes] = size(Phen); % chromosomes and genes number
ObjV=zeros(Nchrom,1); 

for i=1:Nchrom % for each chromosome
    paramSp.Exp_subFrameLength=Phen(i,1); % first value from Phen - WindowSize
    paramSp.Exp_numBands=Phen(i,2); % second value from Phen - NumBands

    if ~isempty(allObjV)
        idx=find((allObjV(:,1) == paramSp.Exp_subFrameLength).* (allObjV(:,2) == paramSp.Exp_numBands));
    else
        idx=[];
    end

    if ~ isempty(idx)
        ObjV(i)=allObjV(idx,3);
    else
        % train the network
        Acc=trainMyNetwork(netMy, dsTrain,paramSp, NoEP,MBS,0,classVtrain,Frequency,dsInVal,classVtrainVal);
        ObjV(i)=1-Acc; % accuracy
        fprintf(1,"solutie %g: ObJv=%g (SL=%g, NB=%g)\n",i,1-Acc,paramSp.Exp_subFrameLength,paramSp.Exp_numBands)
        allObjV=[allObjV;[paramSp.Exp_subFrameLength,paramSp.Exp_numBands,ObjV(i)]];
    end
end

end


%% Train the network
function [Acc, netMy]=trainMyNetwork(netMy, dsTrain,paramSp, NoEP,MBS,withFig,classVtrain,Frequency,dsInVal,classVtrainVal)

mbqTrain = minibatchqueue(dsTrain,...
    MiniBatchSize=MBS,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" ""], ...
    PartialMiniBatch="discard");

% training options
learnRate = 0.0002;
gradientDecay = 0.5;
squaredGradientDecayFactor = 0.999;

avgGrad= [];
avgSqGrad = [];

if withFig==1
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf]);
    xlabel("Iteration");
    ylabel("Loss");
    grid on;
end

iteration = 0;
start = tic;

%training loop
for epoch = 1:NoEP

    reset(mbqTrain);
    shuffle(mbqTrain);

    % Loop over mini-batches
    while hasdata(mbqTrain)
        iteration = iteration + 1;

        % Read mini-batch of data
        [signalIn,labelIm] = next(mbqTrain);

        % Calculate the loss and gradients
        [modelGrad, lossF] = ...
            dlfeval(@modelGradients,signalIn,labelIm,netMy,paramSp,Frequency);

        [netMy.Learnables,avgGrad,avgSqGrad] = ...
            adamupdate(netMy.Learnables,modelGrad,avgGrad,avgSqGrad, ...
            iteration,learnRate,gradientDecay,squaredGradientDecayFactor);


        % Update the plots of network scores
        if withFig==1
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(lossF))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        elseif withFig==2
            fprintf(1,"epoch %d, iteration %d: loss = %g \n",epoch, iteration,lossF);
        end
    end

end

YPred=[];
reset(dsTrain);
labels=unique(categorical(classVtrain));
Acc=0;i=1;
while(hasdata(dsTrain))
    aux=read(dsTrain);
    signalIn=aux{1};
    YPred(i,1) = evaluateModel(signalIn,netMy,labels,paramSp,Frequency);
    LT=aux{2};
    if double(YPred(i,1))==double(LT), Acc=Acc+1; end
    i=i+1;
end
Acc=Acc/(i-1);

end
