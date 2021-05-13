%   ETI_FS_csv.m : Enhanced Temporal Integration - Feature Set
%
%   Author : Kostis Ioannis - Aris
%
%   Project : MIRex 2018 - Speech/Music Detection
%
%   Functionality : The algorithm imports a list of 
%                   .wav files, from two directories 
%                   dedicated accordingly to speech 
%                   and music waveforms. It then  
%                   computes a number temporaly agreggated  
%                   features based on an initial baseline
%                   feature vector a for each sound waveform.
% 
%   Parameters :      
%   set up       @ timeWindows
%                @ hop
%                @ numOfIntegratedWindows                
%
%                Each imported .wav can be segmented
%                in a number of windows time with
%                duration of @timeWindow sec. An overlap
%                can be set as a @hop% percentage 
%                of a time window. After that, we team
%                @numOfIntegratedWindows successive 
%                windows and perform a temporal integration
%                of the baseline features of each team, thus creating
%                new statistical features from the original ones.
%
%                For example : Let's assume that the ROLL-OFF FREQUENCY is
%                              one feature of a time window. Just after the
%                              temporal integration of a number of windows,
%                              the initial feature will now be maped to
%                              some new statistical features representing
%                              its statistical distribution among the
%                              teamed up textures. Some of the new added 
%                              statistical features are :
%                              -  Mean Value
%                              -  Standard Deviation
%                              -  Mean Crossing Rate
%                              -  Mean Absolute Sequential Difference 
%                              -  Flatness 
%
%   Purpose : Build a well structured dataset and
%             export it into dataset_STI.csv  for 
%             future classification model training 
%             and testing. 
%
%   Input 
%   .wav  : Set the right path for the music and
%           speech directories at lines 57 and 58
%           accordingly.

clear all;
close all;

music_samples = dir('music_wav/*.wav');
speech_samples = dir('speech_wav/*.wav');
 
fid = fopen('dataset_ETI.csv','w'); 

header = {  'rms_mean','rms_std','rms_mcr','rms_masd','rms_flat'...
            'zerocross_mean','zerocross_std','zerocross_mcr','zerocross_masd','zerocross_flat'...
            'roll-off_mean','roll-off_std','roll-off_mcr','roll-off_masd','roll-off_flat'...
            'centroid_mean','centroid_std','centroid_mcr','centroid_masd','centroid_flat'...
            'spread_mean','spread_std','spread_mcr','spread_masd','spread_flat'...
            'kurtosis_mean','kurtosis_std','kurtosis_mcr','kurtosis_masd','kurtosis_flat'...
            'flatness_mean','flatness_std','fltaness_mcr', 'fltaness_masd','fltaness_flat'...
            'skewness_mean','skewness_std','skewness_mcr','skewness_masd','skewness_flat'...
            'mfcc1_mean','mfcc1_std','mfcc1_mcr','mfcc1_masd','mfcc1_flat'...
            'mfcc2_mean','mfcc2_std','mfcc2_mcr','mfcc2_masd','mfcc2_flat'...
            'mfcc3_mean','mfcc3_std','mfcc3_mcr','mfcc3_masd','mfcc3_flat'...
            'mfcc4_mean','mfcc4_std','mfcc4_mcr','mfcc4_masd','mfcc4_flat'...
            'mfcc5_mean','mfcc5_std','mfcc5_mcr','mfcc5_masd','mfcc5_flat'...
            'mfcc6_mean','mfcc6_std','mfcc6_mcr','mfcc6_masd','mfcc6_flat'...
            'mfcc7_mean','mfcc7_std','mfcc7_mcr','mfcc7_masd','mfcc7_flat'...
            'mfcc8_mean','mfcc8_std','mfcc8_mcr','mfcc8_masd','mfcc8_flat'...
            'mfcc9_mean','mfcc9_std','mfcc9_mcr','mfcc9_masd','mfcc9_flat'...
            'mfcc10_mean','mfcc10_std','mfcc10_mcr','mfcc10_masd','mfcc10_flat'...
            'mfcc11_mean','mfcc11_std','mfcc11_mcr','mfcc11_masd','mfcc11_flat'...
            'mfcc12_mean','mfcc12_std','mfcc12_mcr','mfcc12_masd','mfcc12_flat'...
            'mfcc13_mean','mfcc13_std','mfcc13_mcr','mfcc13_masd','mfcc13_flat'...
            'class'};
        
fprintf(fid,'%s,',header{1,1:end-1});
fprintf(fid, '%s\n', header{1,end});

class ={'music','speech'};

timeWindow = 0.05; % window in time (sec)
hop = 100; % overlap percentage, referes to window begin
           % ex. : 50% means next temporal window will start 
           % on the half of previous
numOfIntegratedWindows = 25; % timeWindow * numOfIntegratedWindows = temporal integration duration (sec)
                            % In fact a team of 25 successive temporal windows will be an entry 
                            % to our dataset.
                            
% -------------------------  MUSIC .WAVs --------------------------------
for i=1:length(music_samples)
    
    fileName = strcat('music_wav/',music_samples(i).name);
    music = miraudio(fileName,'Frame',timeWindow,'s',hop,'%');
   
    rms = mirgetdata(mirrms(music),'Frame',timeWindow,'s',hop,'%');
    zerocross = mirgetdata(mirzerocross(music),'Frame',timeWindow,'s',hop,'%');
    rolloff = mirgetdata(mirrolloff(music),'Frame',timeWindow,'s',hop,'%');
    centroid = mirgetdata(mircentroid(music),'Frame',timeWindow,'s',hop,'%');
    spread = mirgetdata(mirspread(music),'Frame',timeWindow,'s',hop,'%');
    kurtosis = mirgetdata(mirkurtosis(music),'Frame',timeWindow,'s',hop,'%');
    flatness = mirgetdata(mirflatness(music),'Frame',timeWindow,'s',hop,'%');
    skewness = mirgetdata(mirskewness(music),'Frame',timeWindow,'s',hop,'%');
    mfcc =  mirgetdata(mirmfcc(music),'Frame',timeWindow,'s',hop,'%');
    
    
    for j = 1:numOfIntegratedWindows:length(zerocross)
        
        texture = j:j+numOfIntegratedWindows-1;
        if sum(texture > length(zerocross)) >= 1
            break;
            %texture = j:length(zerocross);
        end
        
         % ------------ rms ----------------
        
        rms_mean = mean(rms(1,texture));
        rms_std =  std(rms(1,texture));
        rms_mcr = lcr(rms(1,texture),mean(rms(1,texture)));
        rms_masd = 1/length(diff(rms(texture)))*sum(abs(diff(rms(texture))));
        rms_flat = geomean(rms(1,texture))/mean(rms(1,texture));
        
        % ------------ zerocross ----------------
        
        zerocross_mean = mean(zerocross(1,texture));
        zerocross_std =  std(zerocross(1,texture));
        zerocross_mcr = lcr(zerocross(1,texture),mean(zerocross(1,texture))); 
        zerocross_masd = 1/length(diff(zerocross(texture)))*sum(abs(diff(zerocross(texture))));
        zerocross_flat = geomean(zerocross(1,texture))/mean(zerocross(1,texture));
        
        % ------------ rolloff ----------------
        
        rolloff_mean = mean(rolloff(1,texture));
        rolloff_std = std(rolloff(1,texture));
        rolloff_mcr = lcr(rolloff(1,texture),mean(rolloff(1,texture)));
        rolloff_masd = 1/length(diff(rolloff(texture)))*sum(abs(diff(rolloff(texture)))); 
        rolloff_flat = geomean(rolloff(1,texture))/mean(rolloff(1,texture));  
        
        % ------------ centroid ----------------
        
        centroid_mean = mean(centroid(1,texture));
        centroid_std = std(centroid(1,texture));
        centroid_mcr = lcr(centroid(1,texture),mean(centroid(1,texture)));
        centroid_masd = 1/length(diff(centroid(texture)))*sum(abs(diff(centroid(texture))));  
        centroid_flat = geomean(centroid(1,texture))/mean(centroid(1,texture));
        
        % ------------ spread ----------------
        
        spread_mean = mean(spread(1,texture));
        spread_std = std(spread(1,texture));
        spread_mcr = lcr(spread(1,texture),mean(spread(1,texture)));
        spread_masd = 1/length(diff(spread(texture)))*sum(abs(diff(spread(texture))));        
        spread_flat = geomean(spread(1,texture))/mean(spread(1,texture));
        
        % ------------ kurtosis ----------------
        
        kurtosis_mean = mean(kurtosis(1,texture));
        kurtosis_std = std(kurtosis(1,texture));
        kurtosis_mcr = lcr(kurtosis(1,texture),mean(kurtosis(1,texture)));
        kurtosis_masd = 1/length(diff(kurtosis(texture)))*sum(abs(diff(kurtosis(texture))));
        kurtosis_flat = geomean(kurtosis(1,texture))/mean(kurtosis(1,texture));
        
        % ------------ flatness ----------------
        
        flatness_mean = mean(flatness(1,texture));
        flatness_std = std(flatness(1,texture));
        flatness_mcr = lcr(flatness(1,texture),mean(flatness(1,texture)));
        flatness_masd = 1/length(diff(flatness(texture)))*sum(abs(diff(flatness(texture))));
        flatness_flat = geomean(flatness(1,texture))/mean(flatness(1,texture));
        
        % ------------ skewness ----------------
        
        skewness_mean = mean(skewness(1,texture));
        skewness_std = std(skewness(1,texture));
        skewness_mcr = lcr(skewness(1,texture),mean(skewness(1,texture)));
        skewness_masd = 1/length(diff(skewness(texture)))*sum(abs(diff(skewness(texture))));
        skewness_flat = nthroot(prod(skewness(texture)),length(texture))/mean(skewness(1,texture));
        
          % ------------ mfcc1 ----------------
        mfcc1_mean = mean(mfcc(1,texture));
        mfcc1_std =  std(mfcc(1,texture));
        mfcc1_mcr = lcr(mfcc(1,texture),mean(mfcc(1,texture)));
        mfcc1_masd = 1/length(diff(mfcc(1,texture)))*sum(abs(diff(mfcc(1,texture))));
        mfcc1_flat = nthroot(prod(mfcc(1,texture)),length(texture))/mean(mfcc(1,texture));
        % ------------ mfcc2 ----------------
        mfcc2_mean = mean(mfcc(2,texture));
        mfcc2_std =  std(mfcc(2,texture));
        mfcc2_mcr = lcr(mfcc(2,texture),mean(mfcc(2,texture)));
        mfcc2_masd = 1/length(diff(mfcc(2,texture)))*sum(abs(diff(mfcc(2,texture))));
        mfcc2_flat = nthroot(prod(mfcc(2,texture)),length(texture))/mean(mfcc(2,texture));
        % ------------ mfcc3 ----------------
        mfcc3_mean = mean(mfcc(3,texture));
        mfcc3_std =  std(mfcc(3,texture));
        mfcc3_mcr = lcr(mfcc(3,texture),mean(mfcc(3,texture)));
        mfcc3_masd = 1/length(diff(mfcc(3,texture)))*sum(abs(diff(mfcc(3,texture))));
        mfcc3_flat = nthroot(prod(mfcc(3,texture)),length(texture))/mean(mfcc(3,texture));
        % ------------ mfcc4 ----------------
        mfcc4_mean = mean(mfcc(4,texture));
        mfcc4_std =  std(mfcc(4,texture));
        mfcc4_mcr = lcr(mfcc(4,texture),mean(mfcc(4,texture)));
        mfcc4_masd = 1/length(diff(mfcc(4,texture)))*sum(abs(mfcc(4,texture)));
        mfcc4_flat = nthroot(prod(mfcc(4,texture)),length(texture))/mean(mfcc(4,texture));
        % ------------ mfcc5 ----------------
        mfcc5_mean = mean(mfcc(5,texture));
        mfcc5_std =  std(mfcc(5,texture));
        mfcc5_mcr = lcr(mfcc(5,texture),mean(mfcc(5,texture)));
        mfcc5_masd = 1/length(diff(mfcc(5,texture)))*sum(abs(diff(mfcc(5,texture))));
        mfcc5_flat = nthroot(prod(mfcc(5,texture)),length(texture))/mean(mfcc(5,texture));
        % ------------ mfcc6 ----------------
        mfcc6_mean = mean(mfcc(6,texture));
        mfcc6_std =  std(mfcc(6,texture));
        mfcc6_mcr = lcr(mfcc(6,texture),mean(mfcc(6,texture)));
        mfcc6_masd = 1/length(diff(mfcc(6,texture)))*sum(abs(diff(mfcc(6,texture))));
        mfcc6_flat = nthroot(prod(mfcc(6,texture)),length(texture))/mean(mfcc(6,texture));
        % ------------ mfcc7 ----------------
        mfcc7_mean = mean(mfcc(7,texture));
        mfcc7_std =  std(mfcc(7,texture));
        mfcc7_mcr = lcr(mfcc(7,texture),mean(mfcc(7,texture)));
        mfcc7_masd = 1/length(diff(mfcc(7,texture)))*sum(abs(diff(mfcc(7,texture))));
        mfcc7_flat = nthroot(prod(mfcc(7,texture)),length(texture))/mean(mfcc(7,texture));
        % ------------ mfcc8 ----------------
        mfcc8_mean = mean(mfcc(8,texture));
        mfcc8_std =  std(mfcc(8,texture));
        mfcc8_mcr = lcr(mfcc(8,texture),mean(mfcc(8,texture)));
        mfcc8_masd = 1/length(diff(mfcc(8,texture)))*sum(abs(diff(mfcc(8,texture))));
        mfcc8_flat = nthroot(prod(mfcc(8,texture)),length(texture))/mean(mfcc(8,texture));
        % ------------ mfcc9 ----------------
        mfcc9_mean = mean(mfcc(9,texture));
        mfcc9_std =  std(mfcc(9,texture));
        mfcc9_mcr = lcr(mfcc(9,texture),mean(mfcc(9,texture)));
        mfcc9_masd = 1/length(diff(mfcc(9,texture)))*sum(abs(diff(mfcc(9,texture))));
        mfcc9_flat = nthroot(prod(mfcc(9,texture)),length(texture))/mean(mfcc(9,texture));
        % ------------ mfcc10 ----------------
        mfcc10_mean = mean(mfcc(10,texture));
        mfcc10_std =  std(mfcc(10,texture));
        mfcc10_mcr = lcr(mfcc(10,texture),mean(mfcc(10,texture)));
        mfcc10_masd = 1/length(diff(mfcc(10,texture)))*sum(abs(diff(mfcc(10,texture))));
        mfcc10_flat = nthroot(prod(mfcc(10,texture)),length(texture))/mean(mfcc(10,texture));
        % ------------ mfcc11 ----------------
        mfcc11_mean = mean(mfcc(11,texture));
        mfcc11_std =  std(mfcc(11,texture));
        mfcc11_mcr = lcr(mfcc(11,texture),mean(mfcc(11,texture)));
        mfcc11_masd = 1/length(diff(mfcc(11,texture)))*sum(abs(diff(mfcc(11,texture))));
        mfcc11_flat = nthroot(prod(mfcc(11,texture)),length(texture))/mean(mfcc(11,texture));
        % ------------ mfcc12 ----------------
        mfcc12_mean = mean(mfcc(12,texture));
        mfcc12_std =  std(mfcc(12,texture));
        mfcc12_mcr = lcr(mfcc(12,texture),mean(mfcc(12,texture)));
        mfcc12_masd = 1/length(diff(mfcc(12,texture)))*sum(abs(diff(mfcc(12,texture))));
        mfcc12_flat = nthroot(prod(mfcc(12,texture)),length(texture))/mean(mfcc(12,texture));
        % ------------ mfcc13 ----------------
        mfcc13_mean = mean(mfcc(13,texture));
        mfcc13_std =  std(mfcc(13,texture));
        mfcc13_mcr = lcr(mfcc(13,texture),mean(mfcc(13,texture)));
        mfcc13_masd = 1/length(diff(mfcc(13,texture)))*sum(abs(diff(mfcc(13,texture))));
        mfcc13_flat = nthroot(prod(mfcc(13,texture)),length(texture))/mean(mfcc(13,texture));
        
        
        row = [     rms_mean,rms_std,rms_mcr,rms_masd,rms_flat...
                    zerocross_mean,zerocross_std,zerocross_mcr,zerocross_masd,zerocross_flat...
                    rolloff_mean,rolloff_std,rolloff_mcr,rolloff_masd,rolloff_flat...
                    centroid_mean,centroid_std,centroid_mcr,centroid_masd,centroid_flat...
                    spread_mean,spread_std,spread_mcr,spread_masd,spread_flat...
                    kurtosis_mean,kurtosis_std,kurtosis_mcr,kurtosis_masd,kurtosis_flat...
                    flatness_mean,flatness_std,flatness_mcr,flatness_masd,flatness_flat...
                    skewness_mean,skewness_std,skewness_mcr,skewness_masd,skewness_flat...
                    mfcc1_mean,mfcc1_std,mfcc1_mcr,mfcc1_masd,mfcc1_flat...
                    mfcc2_mean,mfcc2_std,mfcc2_mcr,mfcc2_masd,mfcc2_flat...
                    mfcc3_mean,mfcc3_std,mfcc3_mcr,mfcc3_masd,mfcc3_flat...
                    mfcc4_mean,mfcc4_std,mfcc4_mcr,mfcc4_masd,mfcc4_flat...
                    mfcc5_mean,mfcc5_std,mfcc5_mcr,mfcc5_masd,mfcc5_flat...
                    mfcc6_mean,mfcc6_std,mfcc6_mcr,mfcc6_masd,mfcc6_flat...
                    mfcc7_mean,mfcc7_std,mfcc7_mcr,mfcc7_masd,mfcc7_flat...
                    mfcc8_mean,mfcc8_std,mfcc8_mcr,mfcc8_masd,mfcc8_flat...
                    mfcc9_mean,mfcc9_std,mfcc9_mcr,mfcc9_masd,mfcc9_flat...
                    mfcc10_mean,mfcc10_std,mfcc10_mcr,mfcc10_masd,mfcc10_flat...
                    mfcc11_mean,mfcc11_std,mfcc11_mcr,mfcc11_masd,mfcc11_flat...
                    mfcc12_mean,mfcc12_std,mfcc12_mcr,mfcc12_masd,mfcc12_flat...
                    mfcc13_mean,mfcc13_std,mfcc13_mcr,mfcc13_masd,mfcc13_flat...
                    ];
            
    
        for jj=1:length(row)
           fprintf(fid,'%f,',row(jj)); 
        end
        fprintf(fid,'%s\n',char(class{1}));
    end
    
    
end

% -------------------------  SPEECH .WAVs --------------------------------
for i=1:length(speech_samples)
   
    fileName = strcat('speech_wav/',speech_samples(i).name);
    speech = miraudio(fileName,'Frame',timeWindow,'s',hop,'%');
    
    rms = mirgetdata( mirrms(speech),'Frame',timeWindow,'s',hop,'%');
    zerocross = mirgetdata( mirzerocross(speech),'Frame',timeWindow,'s',hop,'%');
    rolloff = mirgetdata(mirrolloff(speech),'Frame',timeWindow,'s',hop,'%');
    spread = mirgetdata(mirspread(speech),'Frame',timeWindow,'s',hop,'%');
    kurtosis = mirgetdata(mirkurtosis(speech),'Frame',timeWindow,'s',hop,'%');
    flatness = mirgetdata(mirflatness(speech),'Frame',timeWindow,'s',hop,'%');
    skewness = mirgetdata(mirskewness(speech),'Frame',timeWindow,'s',hop,'%');
    mfcc =  mirgetdata(mirmfcc(speech),'Frame',timeWindow,'s',hop,'%');
    
      for j = 1:numOfIntegratedWindows:length(zerocross)
          
        texture = j:j+numOfIntegratedWindows-1;
        if sum(texture > length(zerocross)) >= 1
            break;
            %texture = j:length(zerocross);
        end
        
          % ------------ rms ----------------
        
        rms_mean = mean(rms(1,texture));
        rms_std =  std(rms(1,texture));
        rms_mcr = lcr(rms(1,texture),mean(rms(1,texture)));
        rms_masd = 1/length(diff(rms(texture)))*sum(abs(diff(rms(texture))));
        rms_flat = geomean(rms(1,texture))/mean(rms(1,texture));
        
        % ------------ zerocross ----------------
        
        zerocross_mean = mean(zerocross(1,texture));
        zerocross_std =  std(zerocross(1,texture));
        zerocross_mcr = lcr(zerocross(1,texture),mean(zerocross(1,texture))); 
        zerocross_masd = 1/length(diff(zerocross(texture)))*sum(abs(diff(zerocross(texture))));
        zerocross_flat = geomean(zerocross(1,texture))/mean(zerocross(1,texture));
        
        % ------------ rolloff ----------------
        
        rolloff_mean = mean(rolloff(1,texture));
        rolloff_std = std(rolloff(1,texture));
        rolloff_mcr = lcr(rolloff(1,texture),mean(rolloff(1,texture)));
        rolloff_masd = 1/length(diff(rolloff(texture)))*sum(abs(diff(rolloff(texture)))); 
        rolloff_flat = geomean(rolloff(1,texture))/mean(rolloff(1,texture));  
        
        % ------------ centroid ----------------
        
        centroid_mean = mean(centroid(1,texture));
        centroid_std = std(centroid(1,texture));
        centroid_mcr = lcr(centroid(1,texture),mean(centroid(1,texture)));
        centroid_masd = 1/length(diff(centroid(texture)))*sum(abs(diff(centroid(texture)))); 
        centroid_flat = geomean(centroid(1,texture))/mean(centroid(1,texture));
        
        % ------------ spread ----------------
        
        spread_mean = mean(spread(1,texture));
        spread_std = std(spread(1,texture));
        spread_mcr = lcr(spread(1,texture),mean(spread(1,texture)));
        spread_masd = 1/length(diff(spread(texture)))*sum(abs(diff(spread(texture))));        
        spread_flat = geomean(spread(1,texture))/mean(spread(1,texture));
        
        % ------------ kurtosis ----------------
        
        kurtosis_mean = mean(kurtosis(1,texture));
        kurtosis_std = std(kurtosis(1,texture));
        kurtosis_mcr = lcr(kurtosis(1,texture),mean(kurtosis(1,texture)));
        kurtosis_masd = 1/length(diff(kurtosis(texture)))*sum(abs(diff(kurtosis(texture))));
        kurtosis_flat = geomean(kurtosis(1,texture))/mean(kurtosis(1,texture));
        
        % ------------ flatness ----------------
        
        flatness_mean = mean(flatness(1,texture));
        flatness_std = std(flatness(1,texture));
        flatness_mcr = lcr(flatness(1,texture),mean(flatness(1,texture)));
        flatness_masd = 1/length(diff(flatness(texture)))*sum(abs(diff(flatness(texture))));
        flatness_flat = geomean(flatness(1,texture))/mean(flatness(1,texture));
        
         % ------------ skewness ----------------
        
        skewness_mean = mean(skewness(1,texture));
        skewness_std = std(skewness(1,texture));
        skewness_mcr = lcr(skewness(1,texture),mean(skewness(1,texture)));
        skewness_masd = 1/length(diff(skewness(texture)))*sum(abs(diff(skewness(texture))));
        skewness_flat = nthroot(prod(skewness(texture)),length(texture))/mean(skewness(1,texture));
        
          % ------------ mfcc1 ----------------
        mfcc1_mean = mean(mfcc(1,texture));
        mfcc1_std =  std(mfcc(1,texture));
        mfcc1_mcr = lcr(mfcc(1,texture),mean(mfcc(1,texture)));
        mfcc1_masd = 1/length(diff(mfcc(1,texture)))*sum(abs(diff(mfcc(1,texture))));
        mfcc1_flat = nthroot(prod(mfcc(1,texture)),length(texture))/mean(mfcc(1,texture));
        % ------------ mfcc2 ----------------
        mfcc2_mean = mean(mfcc(2,texture));
        mfcc2_std =  std(mfcc(2,texture));
        mfcc2_mcr = lcr(mfcc(2,texture),mean(mfcc(2,texture)));
        mfcc2_masd = 1/length(diff(mfcc(2,texture)))*sum(abs(diff(mfcc(2,texture))));
        mfcc2_flat = nthroot(prod(mfcc(2,texture)),length(texture))/mean(mfcc(2,texture));
        % ------------ mfcc3 ----------------
        mfcc3_mean = mean(mfcc(3,texture));
        mfcc3_std =  std(mfcc(3,texture));
        mfcc3_mcr = lcr(mfcc(3,texture),mean(mfcc(3,texture)));
        mfcc3_masd = 1/length(diff(mfcc(3,texture)))*sum(abs(diff(mfcc(3,texture))));
        mfcc3_flat = nthroot(prod(mfcc(3,texture)),length(texture))/mean(mfcc(3,texture));
        % ------------ mfcc4 ----------------
        mfcc4_mean = mean(mfcc(4,texture));
        mfcc4_std =  std(mfcc(4,texture));
        mfcc4_mcr = lcr(mfcc(4,texture),mean(mfcc(4,texture)));
        mfcc4_masd = 1/length(diff(mfcc(4,texture)))*sum(abs(mfcc(4,texture)));
        mfcc4_flat = nthroot(prod(mfcc(4,texture)),length(texture))/mean(mfcc(4,texture));
        % ------------ mfcc5 ----------------
        mfcc5_mean = mean(mfcc(5,texture));
        mfcc5_std =  std(mfcc(5,texture));
        mfcc5_mcr = lcr(mfcc(5,texture),mean(mfcc(5,texture)));
        mfcc5_masd = 1/length(diff(mfcc(5,texture)))*sum(abs(diff(mfcc(5,texture))));
        mfcc5_flat = nthroot(prod(mfcc(5,texture)),length(texture))/mean(mfcc(5,texture));
        % ------------ mfcc6 ----------------
        mfcc6_mean = mean(mfcc(6,texture));
        mfcc6_std =  std(mfcc(6,texture));
        mfcc6_mcr = lcr(mfcc(6,texture),mean(mfcc(6,texture)));
        mfcc6_masd = 1/length(diff(mfcc(6,texture)))*sum(abs(diff(mfcc(6,texture))));
        mfcc6_flat = nthroot(prod(mfcc(6,texture)),length(texture))/mean(mfcc(6,texture));
        % ------------ mfcc7 ----------------
        mfcc7_mean = mean(mfcc(7,texture));
        mfcc7_std =  std(mfcc(7,texture));
        mfcc7_mcr = lcr(mfcc(7,texture),mean(mfcc(7,texture)));
        mfcc7_masd = 1/length(diff(mfcc(7,texture)))*sum(abs(diff(mfcc(7,texture))));
        mfcc7_flat = nthroot(prod(mfcc(7,texture)),length(texture))/mean(mfcc(7,texture));
        % ------------ mfcc8 ----------------
        mfcc8_mean = mean(mfcc(8,texture));
        mfcc8_std =  std(mfcc(8,texture));
        mfcc8_mcr = lcr(mfcc(8,texture),mean(mfcc(8,texture)));
        mfcc8_masd = 1/length(diff(mfcc(8,texture)))*sum(abs(diff(mfcc(8,texture))));
        mfcc8_flat = nthroot(prod(mfcc(8,texture)),length(texture))/mean(mfcc(8,texture));
        % ------------ mfcc9 ----------------
        mfcc9_mean = mean(mfcc(9,texture));
        mfcc9_std =  std(mfcc(9,texture));
        mfcc9_mcr = lcr(mfcc(9,texture),mean(mfcc(9,texture)));
        mfcc9_masd = 1/length(diff(mfcc(9,texture)))*sum(abs(diff(mfcc(9,texture))));
        mfcc9_flat = nthroot(prod(mfcc(9,texture)),length(texture))/mean(mfcc(9,texture));
        % ------------ mfcc10 ----------------
        mfcc10_mean = mean(mfcc(10,texture));
        mfcc10_std =  std(mfcc(10,texture));
        mfcc10_mcr = lcr(mfcc(10,texture),mean(mfcc(10,texture)));
        mfcc10_masd = 1/length(diff(mfcc(10,texture)))*sum(abs(diff(mfcc(10,texture))));
        mfcc10_flat = nthroot(prod(mfcc(10,texture)),length(texture))/mean(mfcc(10,texture));
        % ------------ mfcc11 ----------------
        mfcc11_mean = mean(mfcc(11,texture));
        mfcc11_std =  std(mfcc(11,texture));
        mfcc11_mcr = lcr(mfcc(11,texture),mean(mfcc(11,texture)));
        mfcc11_masd = 1/length(diff(mfcc(11,texture)))*sum(abs(diff(mfcc(11,texture))));
        mfcc11_flat = nthroot(prod(mfcc(11,texture)),length(texture))/mean(mfcc(11,texture));
        % ------------ mfcc12 ----------------
        mfcc12_mean = mean(mfcc(12,texture));
        mfcc12_std =  std(mfcc(12,texture));
        mfcc12_mcr = lcr(mfcc(12,texture),mean(mfcc(12,texture)));
        mfcc12_masd = 1/length(diff(mfcc(12,texture)))*sum(abs(diff(mfcc(12,texture))));
        mfcc12_flat = nthroot(prod(mfcc(12,texture)),length(texture))/mean(mfcc(12,texture));
        % ------------ mfcc13 ----------------
        mfcc13_mean = mean(mfcc(13,texture));
        mfcc13_std =  std(mfcc(13,texture));
        mfcc13_mcr = lcr(mfcc(13,texture),mean(mfcc(13,texture)));
        mfcc13_masd = 1/length(diff(mfcc(13,texture)))*sum(abs(diff(mfcc(13,texture))));
        mfcc13_flat = nthroot(prod(mfcc(13,texture)),length(texture))/mean(mfcc(13,texture));
                
        row = [     rms_mean,rms_std,rms_mcr,rms_masd,rms_flat...
                    zerocross_mean,zerocross_std,zerocross_mcr,zerocross_masd,zerocross_flat...
                    rolloff_mean,rolloff_std,rolloff_mcr,rolloff_masd,rolloff_flat...
                    centroid_mean,centroid_std,centroid_mcr,centroid_masd,centroid_flat...
                    spread_mean,spread_std,spread_mcr,spread_masd,spread_flat...
                    kurtosis_mean,kurtosis_std,kurtosis_mcr,kurtosis_masd,kurtosis_flat...
                    flatness_mean,flatness_std,flatness_mcr,flatness_masd,flatness_flat...
                    skewness_mean,skewness_std,skewness_mcr,skewness_masd,skewness_flat...
                    mfcc1_mean,mfcc1_std,mfcc1_mcr,mfcc1_masd,mfcc1_flat...
                    mfcc2_mean,mfcc2_std,mfcc2_mcr,mfcc2_masd,mfcc2_flat...
                    mfcc3_mean,mfcc3_std,mfcc3_mcr,mfcc3_masd,mfcc3_flat...
                    mfcc4_mean,mfcc4_std,mfcc4_mcr,mfcc4_masd,mfcc4_flat...
                    mfcc5_mean,mfcc5_std,mfcc5_mcr,mfcc5_masd,mfcc5_flat...
                    mfcc6_mean,mfcc6_std,mfcc6_mcr,mfcc6_masd,mfcc6_flat...
                    mfcc7_mean,mfcc7_std,mfcc7_mcr,mfcc7_masd,mfcc7_flat...
                    mfcc8_mean,mfcc8_std,mfcc8_mcr,mfcc8_masd,mfcc8_flat...
                    mfcc9_mean,mfcc9_std,mfcc9_mcr,mfcc9_masd,mfcc9_flat...
                    mfcc10_mean,mfcc10_std,mfcc10_mcr,mfcc10_masd,mfcc10_flat...
                    mfcc11_mean,mfcc11_std,mfcc11_mcr,mfcc11_masd,mfcc11_flat...
                    mfcc12_mean,mfcc12_std,mfcc12_mcr,mfcc12_masd,mfcc12_flat...
                    mfcc13_mean,mfcc13_std,mfcc13_mcr,mfcc13_masd,mfcc13_flat...
                    ];
                            
        for jj=1:length(row)
           fprintf(fid,'%f,',row(jj)); 
        end
         fprintf(fid,'%s\n',char(class{2}));
      end
      
     
end  


fclose(fid);
