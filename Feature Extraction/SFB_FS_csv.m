%   SFB_FS_csv.m : Standard Frame Based - Feature Set
%
%   Author : Kostis Ioannis - Aris
%
%   Project : MIRex 2018 - Speech/Music Detection
%
%   Functionality : The algorithm imports a list of 
%                   .wav files, from two directories 
%                   dedicated accordingly to speech 
%                   and music waveforms. It then  
%                   computes a baseline vector of 
%                   features for each sound waveform.
% 
%   Parameters :      
%   set up       @ timeWindows
%                @ hop
%
%                Each imported .wav can be segmented
%                in a number of windows time with
%                duration of @timeWindow sec. An overlap
%                can be set as a @hop% percentage 
%                of a time window.
%                
%   Purpose : Build a well structured dataset and
%             export it into dataset_SFB.csv  for 
%             future classification model training 
%             and testing.
%
%
%   Input 
%   .wav  : Set the right path for the music and
%           speech directories at lines 38 and 39
%           accordingly

clear all;
close all;

music_samples = dir('music_wav/*.wav');
speech_samples = dir('speech_wav/*.wav');
 
fid = fopen('dataset_SFB.csv','w'); 

header = {  'rms', 'zerocross','rolloff','centroid'...
            'spread', 'kurtosis','flatness','skewness'...
            'mfcc1','mfcc2','mfcc3'...
            'mfcc4','mfcc5','mfcc6'...
            'mfcc7','mfcc8','mfcc9'...
            'mfcc10','mfcc11','mfcc12'...
            'mfcc13','class'};
fprintf(fid,'%s,',header{1,1:end-1});
fprintf(fid, '%s\n', header{1,end});

class ={'music','speech'};

timeWindow = 1.25; % window in time (sec)
hop = 100; % overlap percentage, refers to window beginning
           % ex. 50% means next temporal window will start 
           % on the half of previous

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
    
    
    for j = 1:length(rms)
        row = [ rms(j), zerocross(j), rolloff(j),centroid(j)...
                spread(j), kurtosis(j), flatness(j),skewness(j)...
                mfcc(1,j),mfcc(2,j),mfcc(3,j)...
                mfcc(4,j),mfcc(5,j),mfcc(6,j)...
                mfcc(7,j),mfcc(8,j),mfcc(9,j)...
                mfcc(10,j),mfcc(11,j),mfcc(12,j)...
                mfcc(13,j)];
    
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
    centroid = mirgetdata(mircentroid(speech),'Frame',timeWindow,'s',hop,'%');
    spread = mirgetdata(mirspread(speech),'Frame',timeWindow,'s',hop,'%');
    kurtosis = mirgetdata(mirkurtosis(speech),'Frame',timeWindow,'s',hop,'%');
    flatness = mirgetdata(mirflatness(speech),'Frame',timeWindow,'s',hop,'%');
    skewness = mirgetdata(mirskewness(speech),'Frame',timeWindow,'s',hop,'%');
    mfcc =  mirgetdata(mirmfcc(speech),'Frame',timeWindow,'s',hop,'%');
    
      for j = 1:length(rms)
        row = [ rms(j), zerocross(j), rolloff(j),centroid(j)...
                spread(j), kurtosis(j), flatness(j),skewness(j)...
                mfcc(1,j),mfcc(2,j),mfcc(3,j)...
                mfcc(4,j),mfcc(5,j),mfcc(6,j)...
                mfcc(7,j),mfcc(8,j),mfcc(9,j)...
                mfcc(10,j),mfcc(11,j),mfcc(12,j)...
                mfcc(13,j)];
        for jj=1:length(row)
           fprintf(fid,'%f,',row(jj)); 
        end
         fprintf(fid,'%s\n',char(class{2}));
      end
      
     
end  


fclose(fid);
