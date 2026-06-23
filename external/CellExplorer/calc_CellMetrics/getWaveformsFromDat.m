function spikes = getWaveformsFromDat(spikes,session,varargin)
% Extracts raw waveforms from the binary file. 
% This function is part of CellExplorer: https://cellexplorer.org/
%
% INPUTS
% Spikes struct:            https://cellexplorer.org/datastructure/data-structure-and-format/#spikes
% session metadata struct:  https://cellexplorer.org/datastructure/data-structure-and-format/#session-metadata
%
% Last edited: 21-12-2020

% Loading preferences
preferences = preferences_ProcessCellMetrics(session);

p = inputParser;
addParameter(p,'unitsToProcess',1:size(spikes.times,2), @isnumeric);
addParameter(p,'nPull',preferences.waveform.nPull, @isnumeric); % number of spikes to pull out (default: 600)
addParameter(p,'wfWin_sec', preferences.waveform.wfWin_sec, @isnumeric); % Larger size of waveform windows for filterning. total width in seconds (default: 0.004)
addParameter(p,'wfWinKeep', preferences.waveform.wfWinKeep, @isnumeric); % half width in seconds (default: 0.0008)
addParameter(p,'showWaveforms', true, @islogical);
addParameter(p,'filtFreq',[500,8000], @isnumeric); % Band pass filter (default: 500Hz - 8000Hz)
addParameter(p,'keepWaveforms_filt', false, @islogical); % Keep all extracted filtered waveforms
addParameter(p,'keepWaveforms_raw', false, @islogical); % Keep all extracted raw waveforms
addParameter(p,'saveFig', false, @islogical); % Save figure with data
addParameter(p,'saveMat', true, @islogical); % Save figure with data
addParameter(p,'extraLabel', '', @ischar); % Extra labels in figures
addParameter(p,'getBadChannelsFromDat', true, @islogical); % Determining any extra bad channels from noiselevel of .dat file
addParameter(p,'restrictTime',[],@isnumeric); %[lower upper] bounds for time to restrict spikes.ts to, default ignored

parse(p,varargin{:})

wfWin_sec = p.Results.wfWin_sec;
wfWinKeep = p.Results.wfWinKeep;
keepWaveforms_filt = p.Results.keepWaveforms_filt;
keepWaveforms_raw = p.Results.keepWaveforms_raw;
restrictTime = p.Results.restrictTime;
params = p.Results;

% Loading session struct into separate parameters
basepath = session.general.basePath;
basename = session.general.name;
LSB = session.extracellular.leastSignificantBit;
nChannels = session.extracellular.nChannels;
sr = session.extracellular.sr;
electrodeGroups = session.extracellular.electrodeGroups.channels;
nElectrodeGroups = session.extracellular.nElectrodeGroups;
if isfield(session.extracellular,'fileName') && ~isempty(session.extracellular.fileName)
    fileNameRaw = session.extracellular.fileName;
else
    fileNameRaw = [basename '.dat'];
end
if isempty(fileparts(fileNameRaw)) % this is not the full directory
    datFile = fullfile(basepath,fileNameRaw);
else
    datFile = fileNameRaw;
end
try
    precision = session.extracellular.precision;
catch
    precision = 'int16';
end

timerVal = tic;

if params.filtFreq(2) > sr/2
    params.filtFreq(2) = sr/2-1;
end
% Determining any extra bad channels from noiselevel of .dat file
if params.getBadChannelsFromDat
    try
        session = getBadChannelsFromDat(session,'filtFreq',params.filtFreq,'saveMat',params.saveMat);
    end
end

% Removing channels marked as Bad in session struct
% bad_channels = get_bad_channels(session);
try
    bad_channels = session.channelTags.Bad.channels;
catch
    bad_channels=[];
end
if ~isempty(bad_channels)
    badChannels_message = ['Bad channels detected: ' num2str(bad_channels)];
else
    badChannels_message = 'No bad channels detected. ';
end

% Removing channels that does not exist in SpkGrps
if isfield(session.extracellular,'spikeGroups')
    bad_channels = [bad_channels,setdiff([electrodeGroups{:}],[session.extracellular.spikeGroups.channels{:}])];
end

if isempty(bad_channels)
    goodChannels = 1:nChannels;
else
    goodChannels = setdiff(1:nChannels,bad_channels);
end
nGoodChannels = length(goodChannels);

int_gt_0 = @(n,sr) (isempty(n)) || (n <= 0 ) || (n >= sr/2) || isnan(n);

if int_gt_0(params.filtFreq(1),sr) && ~int_gt_0(params.filtFreq(1),sr)
    [b1, a1] = butter(3, params.filtFreq(2)/sr*2, 'low');
    filter_message = ['Lowpass filter applied: ' num2str(params.filtFreq(2)),' Hz. '];
elseif int_gt_0(params.filtFreq(2),sr) && ~int_gt_0(params.filtFreq(1),sr)
    [b1, a1] = butter(3, params.filtFreq(1)/sr*2, 'high');
    filter_message = ['Highpass filter applied: ' num2str(params.filtFreq(1)),' Hz. '];
else
    [b1, a1] = butter(3, [params.filtFreq(1),params.filtFreq(2)]/sr*2, 'bandpass');
    filter_message = ['Bandpass filter applied: ', num2str(params.filtFreq(1)) ,' - ',num2str(params.filtFreq(2)),' Hz. '];
end

disp(['Getting waveforms from dat file (nPull=', num2str(params.nPull),'). ', filter_message, badChannels_message])
if params.showWaveforms
    fig1 = figure('Name', ['Getting waveforms for ' basename],'NumberTitle', 'off','position',[100,100,1000,800]);
    movegui('center');
end

wfWin = round((wfWin_sec * sr)/2);
window_interval = wfWin-ceil(wfWinKeep*sr):wfWin-1+ceil(wfWinKeep*sr); % +- 0.8 ms of waveform
window_interval2 = wfWin-ceil(1.5*wfWinKeep*sr):wfWin-1+ceil(1.5*wfWinKeep*sr); % +- 1.20 ms of waveform
t1 = toc(timerVal);
[waveformSource,duration,waveformSourceLabel] = initializeWaveformSource(datFile,basepath,basename,nChannels,sr,precision);

% Fit exponential
g = fittype('a*exp(-x/b)+c','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'});

for i = 1:length(params.unitsToProcess)
    ii = params.unitsToProcess(i);
    t1 = toc(timerVal);
    if ~isempty(restrictTime)
        tempRestrictedIdx = find((spikes.times{ii}>=restrictTime(1))&(spikes.times{ii}<restrictTime(2)));
        if isfield(spikes,'ts')
            tempRestricted = spikes.ts{ii}(tempRestrictedIdx);
            spkTmp = tempRestricted(find(tempRestricted./sr > wfWin_sec/1.8 & tempRestricted./sr < duration-wfWin_sec/1.8));
        else
            tempRestricted = spikes.times{ii}(tempRestrictedIdx);
            spkTmp = round(sr * tempRestricted(find(tempRestricted > wfWin_sec/1.8 & tempRestricted < duration-wfWin_sec/1.8)));
        end
    else
        if isfield(spikes,'ts')
            spkTmp = spikes.ts{ii}(find(spikes.ts{ii}./sr > wfWin_sec/1.8 & spikes.ts{ii}./sr < duration-wfWin_sec/1.8));
        else
            spkTmp = round(sr * spikes.times{ii}(find(spikes.times{ii} > wfWin_sec/1.8 & spikes.times{ii} < duration-wfWin_sec/1.8)));
        end
    end
    
    if length(spkTmp) > params.nPull
        spkTmp = spkTmp(randperm(length(spkTmp)));
        spkTmp = sort(spkTmp(1:params.nPull));
    end
    spkTmp = spkTmp(:);
%     % Determines the maximum waveform channel from 100 waveforms across all good channels
%     startIndicies1 = (spkTmp(1:min(100,length(spkTmp))) - wfWin)*nChannels+1;
%     stopIndicies1 =  (spkTmp(1:min(100,length(spkTmp))) + wfWin)*nChannels;
%     X1 = cumsum(accumarray(cumsum([1;stopIndicies1(:)-startIndicies1(:)+1]),[startIndicies1(:);0]-[0;stopIndicies1(:)]-1)+1);
%     wf = LSB * mean(reshape(double(rawData.Data(X1(1:end-1))),nChannels,(wfWin*2),[]),3);
%     wfF2 = zeros((wfWin * 2),nGoodChannels);
%     for jj = 1 : nGoodChannels
%         wfF2(:,jj) = filtfilt(b1, a1, wf(goodChannels(jj),:));
%     end
    
    % Pulls the waveforms from all channels from the dat
    [wf, spkTmp] = extractWaveformsFromSource(spkTmp,wfWin,nChannels,LSB,waveformSource);
    if isempty(spkTmp)
        warning('No spikes remained for waveform extraction for unit %d after applying file-boundary constraints.',ii)
        spikes.rawWaveform{ii} = nan(1,length(window_interval));
        spikes.rawWaveform_std{ii} = nan(1,length(window_interval));
        spikes.filtWaveform{ii} = nan(1,length(window_interval));
        spikes.filtWaveform_std{ii} = nan(1,length(window_interval));
        spikes.rawWaveform_all{ii} = nan(nChannels,length(window_interval2));
        spikes.filtWaveform_all{ii} = nan(nChannels,length(window_interval2));
        spikes.timeWaveform{ii} = ([-ceil(wfWinKeep*sr)*(1/sr):1/sr:(ceil(wfWinKeep*sr)-1)*(1/sr)])*1000;
        spikes.timeWaveform_all{ii} = ([-ceil(1.5*wfWinKeep*sr)*(1/sr):1/sr:(ceil(1.5*wfWinKeep*sr)-1)*(1/sr)])*1000;
        spikes.peakVoltage(ii) = nan;
        spikes.channels_all{ii} = 1:nChannels;
        spikes.peakVoltage_sorted{ii} = nan(1,nChannels);
        spikes.maxWaveform_all{ii} = nan(1,nChannels);
        spikes.maxWaveformCh1(ii) = nan;
        spikes.maxWaveformCh(ii) = nan;
        spikes.shankID(ii) = nan;
        spikes.peakVoltage_expFitLengthConstant(ii) = nan;
        continue
    end
    wfF = zeros((wfWin * 2),length(spkTmp),nChannels);
    for jjj = 1 : nChannels
        wfF(:,:,jjj) = filtfilt(b1, a1, wf(:,:,jjj));
    end
    wfF = permute(wfF,[3,1,2]);
    
    for jjj = 1 : nChannels
        wf(:,:,jjj) = detrend(wf(:,:,jjj));
    end    
    
    wf = permute(wf,[3,1,2]);
    wfF2 = mean(wfF(goodChannels,:,:),3)';
    [~, maxWaveformCh1] = max(max(wfF2(window_interval,:))-min(wfF2(window_interval,:)));
    spikes.maxWaveformCh1(ii) = goodChannels(maxWaveformCh1);
    spikes.maxWaveformCh(ii) = spikes.maxWaveformCh1(ii)-1;
    
    % Assigning shankID to the unit
    for jj = 1:nElectrodeGroups
        if any(electrodeGroups{jj} == spikes.maxWaveformCh1(ii))
            spikes.shankID(ii) = jj;
        end
    end

    rawWaveform_all = mean(wf,3);
    spikes.rawWaveform{ii} = rawWaveform_all(spikes.maxWaveformCh1(ii),window_interval);
    rawWaveform_std = std((wf(spikes.maxWaveformCh1(ii),:,:)-mean(wf(spikes.maxWaveformCh1(ii),:,:),3)),0,3);
    filtWaveform_all = mean(wfF,3);
    spikes.filtWaveform{ii} = filtWaveform_all(spikes.maxWaveformCh1(ii),window_interval);
    filtWaveform_std = std((wfF(spikes.maxWaveformCh1(ii),:,:)-mean(wfF(spikes.maxWaveformCh1(ii),:,:),3)),0,3);
    
    spikes.rawWaveform_all{ii} = rawWaveform_all(:,window_interval2);
    spikes.rawWaveform_std{ii} = rawWaveform_std(window_interval);
    spikes.filtWaveform_all{ii} = filtWaveform_all(:,window_interval2);
    spikes.filtWaveform_std{ii} = filtWaveform_std(window_interval);
    spikes.timeWaveform{ii} = ([-ceil(wfWinKeep*sr)*(1/sr):1/sr:(ceil(wfWinKeep*sr)-1)*(1/sr)])*1000;
    spikes.timeWaveform_all{ii} = ([-ceil(1.5*wfWinKeep*sr)*(1/sr):1/sr:(ceil(1.5*wfWinKeep*sr)-1)*(1/sr)])*1000;
    spikes.peakVoltage(ii) = range(spikes.filtWaveform{ii});
    spikes.channels_all{ii} = [1:nChannels];
    
    [B,I] = sort(range(spikes.filtWaveform_all{ii}(goodChannels,:),2),'descend');
    spikes.peakVoltage_sorted{ii} = zeros(1,nChannels);
    spikes.peakVoltage_sorted{ii}(1:length(goodChannels)) = B;
    spikes.maxWaveform_all{ii} = zeros(1,nChannels);
    spikes.maxWaveform_all{ii}(1:length(goodChannels)) = goodChannels(I);

    % keep all filtered waveforms
    if keepWaveforms_filt
        spikes.waveforms.filt{ii} = wfF(:,window_interval,:);
    end
    
    % keep all raw waveforms
    if keepWaveforms_raw
        spikes.waveforms.raw{ii} = wf(:,window_interval,:);
    end
    
    if keepWaveforms_filt || keepWaveforms_raw
       spikes.waveforms.times{ii} = spkTmp/sr;
    end
    
    % Fitting peakVoltage sorted with exponential function with length constant
    nChannelFit = min([16,length(goodChannels),length(electrodeGroups{spikes.shankID(ii)})]);
    x = 1:nChannelFit;
    y = spikes.peakVoltage_sorted{ii}(x);
    if ~isempty(spikes.times{ii})
        f0 = fit(x',y',g,'StartPoint',[spikes.peakVoltage(ii), 5, 5],'Lower',[1, 0.001, 0],'Upper',[5000, 50, 1000]);
        fitCoeffValues = coeffvalues(f0);
        spikes.peakVoltage_expFitLengthConstant(ii) = fitCoeffValues(2);
    else
        spikes.peakVoltage_expFitLengthConstant(ii) = nan;
    end
    % time = ([-ceil(wfWin_sec/2*sr)*(1/sr):1/sr:(ceil(wfWin_sec/2*sr)-1)*(1/sr)])*1000;
    time = [-wfWin_sec/2:1/sr:wfWin_sec/2]*1000;
    time = time(1:size(wfF2,1));
    if params.showWaveforms 
        if ishandle(fig1)
        figure(fig1)
        subplot(5,3,[1,4]), hold off
        plot(time,wfF2), hold on, plot(time,wfF2(:,maxWaveformCh1),'k','linewidth',2), xlabel('Time (ms)'),title('All channels'),ylabel('Average filtered waveforms across channels (\muV)','Interpreter','tex'),hold off
        subplot(5,3,[2,5]), hold off,
        plot(time,permute(wfF(spikes.maxWaveformCh1(ii),:,:),[2,3,1])), hold on
        plot(time,mean(permute(wfF(spikes.maxWaveformCh1(ii),:,:),[2,3,1]),2),'k','linewidth',2),
        title(['Peak channel = ',num2str(spikes.maxWaveformCh1(ii))]),ylabel('Filtered waveforms from peak channel (\muV)','Interpreter','tex'), xlabel('Time (ms)')
        
        subplot(5,3,[7,10]), hold off,
        plot(spikes.timeWaveform{ii},vertcat(spikes.rawWaveform{1:ii})'), hold on
        plot(spikes.timeWaveform{ii},spikes.rawWaveform{ii},'-k','linewidth',1.5), xlabel('Time (ms)'), ylabel('Raw waveforms (\muV)','Interpreter','tex'), xlim([-0.8,0.8])
        subplot(5,3,[8,11]), hold off,
        plot(spikes.timeWaveform{ii},vertcat(spikes.filtWaveform{1:ii})'), hold on
        plot(spikes.timeWaveform{ii},spikes.filtWaveform{ii},'-k','linewidth',1.5), xlabel('Time (ms)'), ylabel('Filtered waveforms (\muV)','Interpreter','tex'), xlim([-0.8,0.8])
        subplot(5,3,3), hold off
        plot(spkTmp/sr,permute(range((wfF(spikes.maxWaveformCh1(ii),window_interval,:)),2),[3,2,1]),'.b')
        ylabel('Amplitude (\muV)','Interpreter','tex'), xlabel('Time (sec)'), title(['Spike amplitudes (nPull=' num2str(params.nPull),')'])
        subplot(5,3,6), hold off
        plot(spikes.peakVoltage_sorted{ii},'.-b'), hold on
        plot(x,fitCoeffValues(1)*exp(-x/fitCoeffValues(2))+fitCoeffValues(3),'r'),
        title(['Length constant (\lambda) = ',num2str(spikes.peakVoltage_expFitLengthConstant(ii),2)],'Interpreter','tex'), xlabel('Sorted channels'), ylabel('Amplitude (\muV)','Interpreter','tex'), xlim([1,nChannelFit])
        subplot(5,3,9), hold on
        plot(spikes.peakVoltage_sorted{ii}), title('Processed units'), xlabel('Sorted channels'), ylabel('Amplitude (\muV)','Interpreter','tex'), xlim([1,nChannelFit])
        subplot(5,3,12), hold off,
        histogram(spikes.peakVoltage_expFitLengthConstant(params.unitsToProcess(1:i)),20), xlabel('Length constant (\lambda)','Interpreter','tex'), ylabel('Occurances'), axis tight
        subplot(5,3,15), hold off,
        histogram(spikes.peakVoltage(params.unitsToProcess(1:i)),20), xlabel('Amplitudes (\muV)','Interpreter','tex'), ylabel('Occurances'), axis tight
        
        subplot(10,3,[28,29]), hold off, title(['Extraction progress for session: ' basename ,' ', params.extraLabel],'interpreter','none')
        rectangle('Position',[0,0,100*i/length(params.unitsToProcess),1],'FaceColor',[0, 0.4470, 0.7410],'EdgeColor',[0, 0.4470, 0.7410] ,'LineWidth',1), xlim([0,100]), ylim([0,1]), set(gca,'xtick',[],'ytick',[])
        xlabel(['Waveforms: ',num2str(i),'/',num2str(length(params.unitsToProcess)),'. ', num2str(round(toc(timerVal)-t1)),' sec/unit, Duration: ', num2str(round(toc(timerVal)/60)), '/', num2str(round(toc(timerVal)/60/i*length(params.unitsToProcess))),' minutes']);
        
        if params.saveFig && ishandle(fig1)
            % Saving figure
            saveFig1.path = 1; saveFig1.fileFormat = 1; saveFig1.save = 1;
            ce_savefigure(fig1,basepath,[basename, '.getWaveformsFromDat_cell_', num2str(params.unitsToProcess(i))],0,saveFig1)
        end
    else
        disp('Canceling waveform extraction...')
        clear wf wfF wf2 wfF2
        clear rawWaveform rawWaveform_std filtWaveform filtWaveform_std
        error('Waveform extraction canceled by user by closing figure window.')
        end
    end
    clear wf wfF wf2 wfF2
end

spikes.processinginfo.params.WaveformsSource = waveformSourceLabel;
spikes.processinginfo.params.WaveformsFiltFreq = params.filtFreq;
spikes.processinginfo.params.Waveforms_nPull = params.nPull;
spikes.processinginfo.params.WaveformsWin_sec = wfWin_sec;
spikes.processinginfo.params.WaveformsWinKeep = wfWinKeep;
spikes.processinginfo.params.WaveformsFilterType = 'butter';
clear rawWaveform rawWaveform_std filtWaveform filtWaveform_std

% Plots
if params.showWaveforms && ishandle(fig1)
    fig1.Name = [basename, ': Waveform extraction complete. ',num2str(i),' cells processed.  ', num2str(round(toc(timerVal)/60)) ' minutes total'];
    
    % Saving a summary figure for all cells
    timestamp = datestr(now, '_dd-mm-yyyy_HH.MM.SS');
    try
        ce_savefigure(fig1,basepath,[basename, '.getWaveformsFromDat' timestamp])
        disp(['getWaveformsFromDat: Summary figure saved to ', fullfile(basepath, 'SummaryFigures', [basename, '.getWaveformsFromDat', timestamp]),'.png'])
    end
end
disp(['Waveform extraction complete. Total duration: ' num2str(round(toc(timerVal)/60)),' minutes'])
end

function [waveformSource,duration,waveformSourceLabel] = initializeWaveformSource(datFile,basepath,basename,nChannels,sr,precision)
sampleBytes = getPrecisionBytes(precision);

if exist(datFile,'file')
    s = dir(datFile);
    waveformSource.mode = 'single';
    waveformSource.datFile = datFile;
    waveformSource.precision = precision;
    duration = s.bytes/(sampleBytes*nChannels*sr);
    waveformSourceLabel = 'dat file';
    return
end

mergePointsFile = fullfile(basepath,[basename,'.MergePoints.events.mat']);
if ~exist(mergePointsFile,'file')
    error(['Binary file missing: ', datFile, newline, 'MergePoints file missing: ', mergePointsFile])
end

mergeData = load(mergePointsFile,'MergePoints');
if ~isfield(mergeData,'MergePoints')
    error('MergePoints file is missing the MergePoints struct: %s',mergePointsFile)
end

MergePoints = mergeData.MergePoints;
if ~isfield(MergePoints,'timestamps_samples') || ~isfield(MergePoints,'foldernames')
    error('MergePoints file is missing timestamps_samples or foldernames: %s',mergePointsFile)
end

starts = double(MergePoints.timestamps_samples(:,1));
stops = double(MergePoints.timestamps_samples(:,2));
foldernames = MergePoints.foldernames;
if isstring(foldernames)
    foldernames = cellstr(foldernames);
end

if numel(foldernames) ~= numel(starts)
    error('Mismatch between MergePoints foldernames and timestamps_samples in %s',mergePointsFile)
end

segments = repmat(struct('foldername','','datFile','','startSample',0,'endSample',0,'nSamples',0,'fileNChannels',0,'dataChannels',[],'excludedChannels',[],'sourceType',''),1,numel(foldernames));
for i = 1:numel(foldernames)
    foldername = localEpochNameFromPath(foldernames{i});
    epochDir = resolveMergePointEpochDir(basepath,foldernames{i});
    datPath = fullfile(epochDir,'amplifier.dat');
    if ~exist(datPath,'file')
        datPath = findOpenEphysContinuousDat(epochDir);
        if isempty(datPath)
            error('Expected amplifier.dat or Open Ephys continuous.dat for MergePoints segment is missing: %s',epochDir)
        end
        sourceType = 'Open Ephys continuous.dat';
    else
        sourceType = 'amplifier.dat';
    end
    fileInfo = dir(datPath);
    expectedSamples = stops(i) - starts(i);
    if expectedSamples <= 0
        error('Invalid MergePoints sample range for %s: start=%d, stop=%d.',foldername,starts(i),stops(i))
    end
    candidateSamples = unique([expectedSamples, expectedSamples + 1, expectedSamples - 1],'stable');
    candidateSamples = candidateSamples(candidateSamples > 0);
    candidateNChannels = fileInfo.bytes ./ (sampleBytes * candidateSamples);
    isIntegerChannelCount = abs(candidateNChannels - round(candidateNChannels)) <= 1e-9;
    isUsableChannelCount = isIntegerChannelCount & round(candidateNChannels) >= nChannels;
    if ~any(isUsableChannelCount)
        error('Could not infer an integer channel count for %s from MergePoints samples (%d) and file size (%d bytes).',datPath,expectedSamples,fileInfo.bytes)
    end
    bestIdx = find(isUsableChannelCount,1,'first');
    expectedSamples = candidateSamples(bestIdx);
    fileNChannels = round(candidateNChannels(bestIdx));
    segments(i).foldername = foldername;
    segments(i).datFile = datPath;
    segments(i).startSample = starts(i);
    segments(i).endSample = starts(i) + expectedSamples;
    segments(i).nSamples = expectedSamples;
    segments(i).fileNChannels = fileNChannels;
    segments(i).dataChannels = 1:nChannels;
    segments(i).excludedChannels = nChannels+1:fileNChannels;
    segments(i).sourceType = sourceType;
end

waveformSource.mode = 'mergepoints';
waveformSource.segments = segments;
waveformSource.precision = precision;
duration = max([segments.endSample])/sr;
sourceTypes = unique({segments.sourceType});
if numel(sourceTypes) == 1 && strcmp(sourceTypes{1},'amplifier.dat')
    waveformSourceLabel = 'MergePoints amplifier.dat files';
elseif numel(sourceTypes) == 1 && strcmp(sourceTypes{1},'Open Ephys continuous.dat')
    waveformSourceLabel = 'MergePoints Open Ephys continuous.dat files';
else
    waveformSourceLabel = 'MergePoints mixed binary files';
end
end

function [wf, spkTmp] = extractWaveformsFromSource(spkTmp,wfWin,nChannels,LSB,waveformSource)
switch waveformSource.mode
    case 'single'
        wf = readWaveformsFromFile(waveformSource.datFile,spkTmp,wfWin,nChannels,1:nChannels,LSB,waveformSource.precision);
    case 'mergepoints'
        [spkTmp,segmentIds] = filterSpikesForSegments(spkTmp,wfWin,waveformSource.segments);
        if isempty(spkTmp)
            wf = [];
            return
        end
        wf = zeros(wfWin*2,length(spkTmp),nChannels);
        uniqueSegments = unique(segmentIds);
        for iSegment = uniqueSegments(:)'
            idx = find(segmentIds == iSegment);
            localSpikes = spkTmp(idx) - waveformSource.segments(iSegment).startSample;
            wf(:,idx,:) = readWaveformsFromFile(waveformSource.segments(iSegment).datFile,localSpikes,wfWin,waveformSource.segments(iSegment).fileNChannels,waveformSource.segments(iSegment).dataChannels,LSB,waveformSource.precision);
        end
    otherwise
        error('Unknown waveform source mode: %s',waveformSource.mode)
end
end

function [spkTmpValid,segmentIds] = filterSpikesForSegments(spkTmp,wfWin,segments)
spkTmpValid = [];
segmentIds = [];
for i = 1:numel(segments)
    inSegment = spkTmp > (segments(i).startSample + wfWin) & spkTmp <= (segments(i).endSample - wfWin);
    if any(inSegment)
        spkTmpValid = [spkTmpValid; spkTmp(inSegment)]; %#ok<AGROW>
        segmentIds = [segmentIds; repmat(i,sum(inSegment),1)]; %#ok<AGROW>
    end
end
[spkTmpValid,order] = sort(spkTmpValid);
segmentIds = segmentIds(order);
end

function wf = readWaveformsFromFile(datFile,spkTmp,wfWin,fileNChannels,dataChannels,LSB,precision)
rawData = memmapfile(datFile,'Format',precision,'writable',false);
startIndicies = (spkTmp - wfWin)*fileNChannels+1;
stopIndicies = (spkTmp + wfWin)*fileNChannels;
X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
wf = LSB * permute(reshape(double(rawData.Data(X(1:end-1))),fileNChannels,(wfWin*2),[]),[2,3,1]);
wf = wf(:,:,dataChannels);
end

function foundFile = findOpenEphysContinuousDat(epochDir)
foundFile = '';
if ~exist(epochDir,'dir')
    return
end

hits = dir(fullfile(epochDir,'**','continuous.dat'));
if isempty(hits)
    return
end

fullPaths = fullfile({hits.folder},{hits.name});
streamFolders = cellfun(@fileparts,fullPaths,'UniformOutput',false);
streamNames = regexp(streamFolders,'[^\\/]+$','match','once');
isLFP = contains(streamNames,'LFP','IgnoreCase',true) | ~cellfun(@isempty,regexp(streamNames,'\.1$','once'));
isMemory = contains(streamNames,'memory_usage','IgnoreCase',true);
fullPaths = fullPaths(~isLFP & ~isMemory);
if isempty(fullPaths)
    return
end

idx = find(contains(fullPaths,'acquisition_board','IgnoreCase',true),1,'first');
if ~isempty(idx)
    foundFile = fullPaths{idx};
    return
end

idx = find(~cellfun(@isempty,regexpi(fullPaths,'ProbeA-AP')),1,'first');
if ~isempty(idx)
    foundFile = fullPaths{idx};
    return
end

idx = find(~cellfun(@isempty,regexpi(fullPaths,'Neuropix.*\.0')),1,'first');
if ~isempty(idx)
    foundFile = fullPaths{idx};
    return
end

idx = find(~cellfun(@isempty,regexpi(fullPaths,'ProbeA')),1,'first');
if ~isempty(idx)
    foundFile = fullPaths{idx};
    return
end

foundFile = fullPaths{1};
end

function epochDir = resolveMergePointEpochDir(basepath,foldername)
localFolderName = localEpochNameFromPath(foldername);
candidates = {
    fullfile(basepath,localFolderName), ...
    fullfile(basepath,char(foldername)), ...
    char(foldername)};
for i = 1:numel(candidates)
    if exist(candidates{i},'dir')
        epochDir = candidates{i};
        return
    end
end
epochDir = fullfile(basepath,localFolderName);
end

function epochName = localEpochNameFromPath(foldername)
foldername = char(foldername);
parts = regexp(strrep(foldername,'\','/'),'[^/]+','match');
if isempty(parts)
    epochName = foldername;
else
    epochName = parts{end};
end
end

function sampleBytes = getPrecisionBytes(precision)
switch lower(precision)
    case {'int16','uint16'}
        sampleBytes = 2;
    case {'int32','uint32','single','float32'}
        sampleBytes = 4;
    case {'int64','uint64','double','float64'}
        sampleBytes = 8;
    case {'int8','uint8','char'}
        sampleBytes = 1;
    otherwise
        error('Unsupported extracellular precision: %s',precision)
end
end
