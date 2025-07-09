function out_NN = evaluateModel(frame,netMy,labels,paramSp,Frequency)   
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
     
       imageIm = dlarray(double(S),"SSCB");
    
       netOutSF = forward(netMy,imageIm);
       [~,idx]=max(extractdata(netOutSF));
       out_NN=labels(idx);
   
end