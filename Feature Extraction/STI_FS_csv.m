%   STI_FS_csv.m : Standard Temporal Integration - Feature Set
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
%                in a number of time windows with
%                duration of @timeWindow sec. An overlap
%                can be set as a @hop% percentage 
%                of a time window. After that, we team up
%                @numOfIntegratedWindows successive 
%                windows and perform a temporal integration
%                of the baseline features of each team, thus creating
%                new statistical features from the original ones.
%
%                For example : Let's assume that the ROLL-OFF FREQUENCY is
%                              a base feature of a time window. Just after the
%                              temporal integration of a number of windows,
%                              the initial feature will now be maped to
%                              some new statistical metrics representing
%                              its statistical distribution among the
%                              teamed up textures. The new added statistical
%                              features are:
%                              -  Mean Value
%                              -  Standard deviation
%                              
%                
%   Purpose : Build a well structured dataset and
%             export it into dataset_STI.csv  for 
%             future classification model training 
%             and testing. 
%
%
%   Input 
%   .wav  : Set the right path for the music and
%           speech directories at lines 56 and 57
%           accordingly

clear all;
close all;

music_samples = dir('music_wav/*.wav');
speech_samples = dir('speech_wav/*.wav');
 
fid = fopen('dataset_STI.csv','w'); 

header = {  'rms_mean','rms_std'...
            'zerocross_mean','zerocross_std'...
            'roll-off_mean','roll-off_std'...
            'centroid_mean','centroid_std'...
            'spread_mean','spread_std'...
            'kurtosis_mean','kurtosis_std'...
            'flatness_mean','flatness_std'...
            'skewness_mean','skewness_std'...
            'mfcc1_mean','mfcc1_std'...
            'mfcc2_mean','mfcc2_std'...
            'mfcc3_mean','mfcc3_std'...
            'mfcc4_mean','mfcc4_std'...
            'mfcc5_mean','mfcc5_std'...
            'mfcc6_mean','mfcc6_std'...
            'mfcc7_mean','mfcc7_std'...
            'mfcc8_mean','mfcc8_std'...
            'mfcc9_mean','mfcc9_std'...
            'mfcc10_mean','mfcc10_std'...
            'mfcc11_mean','mfcc11_std'...
            'mfcc12_mean','mfcc12_std'...
            'mfcc13_mean','mfcc13_std','class'};
          
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
        
        % ------------ zerocross ----------------
        
        zerocross_mean = mean(zerocross(1,texture));
        zerocross_std =  std(zerocross(1,texture));
                      
        % ------------ rolloff ----------------
        
        rolloff_mean = mean(rolloff(1,texture));
        rolloff_std = std(rolloff(1,texture));
        
        % ------------ centroid ----------------
        
        centroid_mean = mean(centroid(1,texture));
        centroid_std = std(centroid(1,texture));
        
        % ------------ spread ----------------
        
        spread_mean = mean(spread(1,texture));
        spread_std = std(spread(1,texture));
        
        % ------------ kurtosis ----------------
        
        kurtosis_mean = mean(kurtosis(1,texture));
        kurtosis_std = std(kurtosis(1,texture));
        
         % ------------ flatness ----------------
        
        flatness_mean = mean(flatness(1,texture));
        flatness_std = std(flatness(1,texture));
        
         % ------------ skewness ----------------
        
        skewness_mean = mean(skewness(1,texture));
        skewness_std = std(skewness(1,texture));
        
        % ------------ mfcc1 ----------------
        mfcc1_mean = mean(mfcc(1,texture));
        mfcc1_std =  std(mfcc(1,texture));
        % ------------ mfcc2 ----------------
        mfcc2_mean = mean(mfcc(2,texture));
        mfcc2_std =  std(mfcc(2,texture));
        % ------------ mfcc3 ----------------
        mfcc3_mean = mean(mfcc(3,texture));
        mfcc3_std =  std(mfcc(3,texture));
        % ------------ mfcc4 ----------------
        mfcc4_mean = mean(mfcc(4,texture));
        mfcc4_std =  std(mfcc(4,texture));
        % ------------ mfcc5 ----------------
        mfcc5_mean = mean(mfcc(5,texture));
        mfcc5_std =  std(mfcc(5,texture));
        % ------------ mfcc6 ----------------
        mfcc6_mean = mean(mfcc(6,texture));
        mfcc6_std =  std(mfcc(6,texture));
        % ------------ mfcc7 ----------------
        mfcc7_mean = mean(mfcc(7,texture));
        mfcc7_std =  std(mfcc(7,texture));
        % ------------ mfcc8 ----------------
        mfcc8_mean = mean(mfcc(8,texture));
        mfcc8_std =  std(mfcc(8,texture));
        % ------------ mfcc9 ----------------
        mfcc9_mean = mean(mfcc(9,texture));
        mfcc9_std =  std(mfcc(9,texture));
        % ------------ mfcc10 ----------------
        mfcc10_mean = mean(mfcc(10,texture));
        mfcc10_std =  std(mfcc(10,texture));
        % ------------ mfcc11 ----------------
        mfcc11_mean = mean(mfcc(11,texture));
        mfcc11_std =  std(mfcc(11,texture));
        % ------------ mfcc12 ----------------
        mfcc12_mean = mean(mfcc(12,texture));
        mfcc12_std =  std(mfcc(12,texture));
        % ------------ mfcc13 ----------------
        mfcc13_mean = mean(mfcc(13,texture));
        mfcc13_std =  std(mfcc(13,texture));
        
        row = [ rms_mean,rms_std...
                zerocross_mean, zerocross_std ...
                rolloff_mean, rolloff_std ...
                centroid_mean, centroid_std ...
                spread_mean, spread_std ...
                kurtosis_mean, kurtosis_std ...
                flatness_mean, flatness_std ...
                skewness_mean, skewness_std ...
                mfcc1_mean, mfcc1_std ...
                mfcc2_mean, mfcc2_std ...
                mfcc3_mean, mfcc3_std ...
                mfcc4_mean, mfcc4_std ...
                mfcc5_mean, mfcc5_std ...
                mfcc6_mean, mfcc6_std ...
                mfcc7_mean, mfcc7_std ...
                mfcc8_mean, mfcc8_std ...
                mfcc9_mean, mfcc9_std ...
                mfcc10_mean, mfcc10_std ...
                mfcc11_mean, mfcc11_std ...
                mfcc12_mean, mfcc12_std ...
                mfcc13_mean, mfcc13_std ...
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
        
          
        % ------------ zerocross ----------------
        
        zerocross_mean = mean(zerocross(1,texture));
        zerocross_std =  std(zerocross(1,texture));
        
        % ------------ rolloff ----------------
        
        rolloff_mean = mean(rolloff(1,texture));
        rolloff_std =  std(rolloff(1,texture));
        
        % ------------ centroid ----------------
        
        centroid_mean = mean(centroid(1,texture));
        centroid_std = std(centroid(1,texture));
        
        % ------------ spread ----------------
        
        spread_mean = mean(spread(1,texture));
        spread_std = std(spread(1,texture));
        
        % ------------ kurtosis ----------------
        
        kurtosis_mean = mean(kurtosis(1,texture));
        kurtosis_std = std(kurtosis(1,texture));
        
         % ------------ flatness ----------------
        
        flatness_mean = mean(flatness(1,texture));
        flatness_std = std(flatness(1,texture));
        
         % ------------ skewness ----------------
        
        skewness_mean = mean(skewness(1,texture));
        skewness_std = std(skewness(1,texture));
        
        
        % ------------ mfcc1 ----------------
        mfcc1_mean = mean(mfcc(1,texture));
        mfcc1_std =  std(mfcc(1,texture));
        % ------------ mfcc2 ----------------
        mfcc2_mean = mean(mfcc(2,texture));
        mfcc2_std =  std(mfcc(2,texture));
        % ------------ mfcc3 ----------------
        mfcc3_mean = mean(mfcc(3,texture));
        mfcc3_std =  std(mfcc(3,texture));        
        % ------------ mfcc4 ----------------
        mfcc4_mean = mean(mfcc(4,texture));
        mfcc4_std =  std(mfcc(4,texture));
        % ------------ mfcc5 ----------------
        mfcc5_mean = mean(mfcc(5,texture));
        mfcc5_std =  std(mfcc(5,texture));
        % ------------ mfcc6 ----------------
        mfcc6_mean = mean(mfcc(6,texture));
        mfcc6_std =  std(mfcc(6,texture));
        % ------------ mfcc7 ----------------
        mfcc7_mean = mean(mfcc(7,texture));
        mfcc7_std =  std(mfcc(7,texture));
        % ------------ mfcc8 ----------------
        mfcc8_mean = mean(mfcc(8,texture));
        mfcc8_std =  std(mfcc(8,texture));
        % ------------ mfcc9 ----------------
        mfcc9_mean = mean(mfcc(9,texture));
        mfcc9_std =  std(mfcc(9,texture));
        % ------------ mfcc10 ----------------
        mfcc10_mean = mean(mfcc(10,texture));
        mfcc10_std =  std(mfcc(10,texture));
        % ------------ mfcc11 ----------------
        mfcc11_mean = mean(mfcc(11,texture));
        mfcc11_std =  std(mfcc(11,texture));
        % ------------ mfcc12 ----------------
        mfcc12_mean = mean(mfcc(12,texture));
        mfcc12_std =  std(mfcc(12,texture));
        % ------------ mfcc13 ----------------
        mfcc13_mean = mean(mfcc(13,texture));
        mfcc13_std =  std(mfcc(13,texture));
        
        
        row = [ rms_mean,rms_std...
                zerocross_mean, zerocross_std ...
                rolloff_mean, rolloff_std ...
                centroid_mean, centroid_std ...
                spread_mean, spread_std ...
                kurtosis_mean, kurtosis_std ...
                flatness_mean, flatness_std ...
                skewness_mean, skewness_std ...
                mfcc1_mean, mfcc1_std ...
                mfcc2_mean, mfcc2_std ...
                mfcc3_mean, mfcc3_std ...
                mfcc4_mean, mfcc4_std ...
                mfcc5_mean, mfcc5_std ...
                mfcc6_mean, mfcc6_std ...
                mfcc7_mean, mfcc7_std ...
                mfcc8_mean, mfcc8_std ...
                mfcc9_mean, mfcc9_std ...
                mfcc10_mean, mfcc10_std ...
                mfcc11_mean, mfcc11_std ...
                mfcc12_mean, mfcc12_std ...
                mfcc13_mean, mfcc13_std ...
                ];
        for jj=1:length(row)
           fprintf(fid,'%f,',row(jj)); 
        end
         fprintf(fid,'%s\n',char(class{2}));
      end
      
     
end  


fclose(fid);
