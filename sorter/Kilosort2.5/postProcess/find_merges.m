function rez = find_merges(rez, flag)
% this function merges clusters based on template correlation
% however, a merge is veto-ed if refractory period violations are introduced

ops = rez.ops;
dt = 1/1000;
mergeTemplateSimilarityThreshold = getOr(ops, 'mergeTemplateSimilarityThreshold', 0.5);
mergeShapeEnable = getOr(ops, 'mergeShapeEnable', false);
mergeShapeMinCorr = getOr(ops, 'mergeShapeMinCorr', 0.8);
mergeShapeExcludeMs = getOr(ops, 'mergeShapeExcludeMs', 2);
mergeShapeWindowMs = getOr(ops, 'mergeShapeWindowMs', 50);

Xsim = rez.simScore; % this is the pairwise similarity score
Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim)); % remove the diagonal of ones

% sort by firing rate first
nspk = accumarray(rez.st3(:,2), 1, [Nk, 1], @sum);
[~, isort] = sort(nspk); % we traverse the set of neurons in ascending order of firing rates

fprintf('initialized spike counts\n')

if ~flag
  % if the flag is off, then no merges are performed
  % this function is then just used to compute cross- and auto- correlograms
   rez.R_CCG = Inf * ones(Nk);
   rez.Q_CCG = Inf * ones(Nk);
   rez.K_CCG = {};
end

for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==isort(j), 1)/ops.fs; % find all spikes from this cluster
    if numel(s1)~=nspk(isort(j))
        fprintf('lost track of spike counts') %this is a check for myself to make sure new cluster are combined correctly into bigger clusters
    end
    % sort all the pairs of this neuron, discarding any that have fewer spikes
    [ccsort, ix] = sort(Xsim(isort(j),:) .* (nspk'>numel(s1)), 'descend');
    ienu = find(ccsort<mergeTemplateSimilarityThreshold, 1) - 1; % find the first pair which has too low of a correlation

    % for all pairs above the configured template similarity threshold
    for k = 1:ienu
        s2 = rez.st3(rez.st3(:,2)==ix(k), 1)/ops.fs; % find the spikes of the pair
        % compute cross-correlograms, refractoriness scores (Qi and rir), and normalization for these scores
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
        Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
        R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes

        if flag
            if Q<.2 && R<.05 % if both refractory criteria are met
                if mergeShapeEnable
                    [K11, ~, ~, ~, ~] = ccg(s1, s1, 500, dt);
                    [K22, ~, ~, ~, ~] = ccg(s2, s2, 500, dt);
                    [passesShapeGate, ccgCorr1, ccgCorr2] = merge_shape_gate( ...
                        K, K11, K22, dt, mergeShapeExcludeMs, mergeShapeWindowMs, mergeShapeMinCorr);
                    if ~passesShapeGate
                        fprintf('shape-vetoed %d into %d (corr %.3f, %.3f) \n', isort(j), ix(k), ccgCorr1, ccgCorr2)
                        continue;
                    end
                end
                i = ix(k);
                % now merge j into i and move on
                rez.st3(rez.st3(:,2)==isort(j),2) = i; % simply overwrite all the spikes of neuron j with i (i>j by construction)
                nspk(i) = nspk(i) + nspk(isort(j)); % update number of spikes for cluster i
                fprintf('merged %d into %d \n', isort(j), i)
                % YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                break; % if a pair is found, we don't need to keep going (we'll revisit this cluster when we get to the merged cluster)
            end
        else
          % sometimes we just want to get the refractory scores and CCG
            rez.R_CCG(isort(j), ix(k)) = R;
            rez.Q_CCG(isort(j), ix(k)) = Q;

            rez.K_CCG{isort(j), ix(k)} = K;
            rez.K_CCG{ix(k), isort(j)} = K(end:-1:1); % the CCG is "antisymmetrical"
        end
    end
end

if ~flag
    rez.R_CCG  = min(rez.R_CCG , rez.R_CCG'); % symmetrize the scores
    rez.Q_CCG  = min(rez.Q_CCG , rez.Q_CCG');
end
end

function [passesShapeGate, corr1, corr2] = merge_shape_gate(K12, K11, K22, dt, excludeMs, windowMs, minCorr)
shape12 = correlogram_shape_vector(K12, dt, excludeMs, windowMs);
shape11 = correlogram_shape_vector(K11, dt, excludeMs, windowMs);
shape22 = correlogram_shape_vector(K22, dt, excludeMs, windowMs);

corr1 = pearson_corr_safe(shape12, shape11);
corr2 = pearson_corr_safe(shape12, shape22);
passesShapeGate = ~isnan(corr1) && ~isnan(corr2) && corr1 >= minCorr && corr2 >= minCorr;
end

function shape = correlogram_shape_vector(K, dt, excludeMs, windowMs)
shape = [];

if isempty(K) || ~isvector(K)
    return;
end

binMs = dt * 1000;
excludeBins = max(0, round(excludeMs / binMs));
windowBins = max(0, round(windowMs / binMs));

if windowBins <= excludeBins
    return;
end

center = (numel(K) + 1) / 2;
if center ~= round(center)
    return;
end

lags = (excludeBins + 1):windowBins;
posIdx = center + lags;
negIdx = center - lags;

if any(posIdx > numel(K)) || any(negIdx < 1)
    return;
end

% Compare only the off-center correlogram shape after collapsing positive and negative lags.
symShape = (double(K(posIdx)) + double(K(negIdx))) / 2;
symShape = conv(symShape(:)', ones(1, 3) / 3, 'same');

shape = zscore_safe(symShape(:));
if isempty(shape)
    return;
end

if any(~isfinite(shape))
    shape = [];
end
end

function out = zscore_safe(x)
out = [];
x = double(x(:));
if isempty(x) || any(~isfinite(x))
    return;
end

mu = mean(x);
sigma = std(x);
if ~isfinite(mu) || ~isfinite(sigma) || sigma <= eps
    return;
end

out = (x - mu) / sigma;
end

function r = pearson_corr_safe(x, y)
r = NaN;
if isempty(x) || isempty(y) || numel(x) ~= numel(y)
    return;
end

x = zscore_safe(x);
y = zscore_safe(y);
if isempty(x) || isempty(y)
    return;
end

r = sum(x .* y) / max(numel(x) - 1, 1);
if ~isfinite(r)
    r = NaN;
end
end
