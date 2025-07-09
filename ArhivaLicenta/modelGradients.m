function [modelGradNet,lossF] = modelGradients(X,labelIm,netMy, paramSp,Frequency)

    
    [nr,nc,ns,ne]=size(X);
    
    for i=1:ne
        frame=X(:,:,:,i); %de ce?
        frame=extractdata(frame);
        
        SL = 2 ^ paramSp.Exp_subFrameLength;
        NB = 2 ^ paramSp.Exp_numBands;
    
        S = melSpectrogram(frame',Frequency, ...
            'Window', hann( SL,'periodic'), ...
            'OverlapLength',  SL/4, ...
            'FFTLength', 2*SL, ... %nu foarte mare ca impart pe subferestre
            'NumBands',  NB, ...
            'FrequencyRange', [0, Frequency/2]); %intervalul in care se face filtrarea frecv din semnal
    
        S = abs(10*log10(S+eps));
    
        S=imresize(S,[227 227]);
    
        Y(:,:,:,i)=S;
    
    
    end
    
    Y=dlarray(Y,"SSCB");
    netOutSF = forward(netMy,Y);
    lossF=crossentropy(netOutSF,labelIm);

    modelGradNet = dlgradient(lossF,netMy.Learnables,'RetainData',true);
    % modelGradExp_subFrameLeng = dlgradient(lossF,paramSp.Exp_subFrameLength,'RetainData',true);
    % modelGradExp_numBands = dlgradient(lossF,paramSp.Exp_numBands,'RetainData',true);

end