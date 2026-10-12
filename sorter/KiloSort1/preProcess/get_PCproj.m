function Us = get_PCproj(S1, row, col, wPCA, maskMaxChans)

[nT, nChan] = size(S1);
nt0 = size(wPCA, 1);
% Match the scaled waveform alignment used by fitTemplates. The original
% 20-sample offset is valid for 61 samples only; retain it at that length.
nBefore = ceil(20 * nt0 / 61);
dt = (0:nt0-1) - nBefore;
if any(row - nBefore < 1 | row + nt0 - nBefore - 1 > nT)
    error('KiloSort:WaveformOutOfBounds', ...
        'Initialization waveform extends beyond its data batch (nt0=%d).', nt0);
end
inds = repmat(row', numel(dt), 1) + repmat(dt', 1, numel(row));

clips = reshape(S1(inds, :), numel(dt), numel(row), nChan);


mask = repmat([1:nChan], [numel(row) 1]) - repmat(col, 1, nChan);
Mask(1,:,:) = abs(mask)<maskMaxChans;

clips = bsxfun(@times, clips , Mask);

Us = wPCA' * reshape(clips, numel(dt), []);
Us = reshape(Us, size(wPCA,2), numel(row), nChan);

Us = permute(Us, [3 2 1]);
