
function predictSongToMIDI(trainedNet, vocab, seedNotes, totalLength, temperature, outputFile)
    if nargin < 5
        temperature = 1.0;
    end
    if nargin < 6
        outputFile = 'predicted_song_from_seed.mid';
    end
    notes = predictSongFromSeed(trainedNet, vocab, seedNotes, totalLength, temperature);
    durationBucket = 0;
    velocityBucket = 4;
    tokens = zeros(totalLength, 3);
    for i = 1:totalLength
        tokens(i, :) = [notes(i), durationBucket, velocityBucket];
    end
    msgArray = [];
    currentTime = 0;
    seedLen = length(seedNotes);
    duration = 0.35;
    for i = 1:seedLen
        note = seedNotes(i);
        velocity = min(127, max(1, velocityBucket*16));
        noteOnStruct = struct();
        noteOnStruct.Type = 'NoteOn';
        noteOnStruct.Note = note;
        noteOnStruct.Velocity = velocity;
        noteOnStruct.Timestamp = currentTime;
        noteOnStruct.Channel = 1;
        noteOnStruct.RawBytes = [144, note, velocity, 0, 0, 0, 0, 0];
        noteOffStruct = struct();
        noteOffStruct.Type = 'NoteOff';
        noteOffStruct.Note = note;
        noteOffStruct.Velocity = 0;
        noteOffStruct.Timestamp = currentTime + duration*0.8;
        noteOffStruct.Channel = 1;
        noteOffStruct.RawBytes = [128, note, 0, 0, 0, 0, 0, 0];
        msgArray = [msgArray; noteOnStruct; noteOffStruct];
        currentTime = currentTime + duration;
    end
    for i = seedLen+1:totalLength
        note = notes(i);
        velocity = min(127, max(1, velocityBucket*16));
        noteOnStruct = struct();
        noteOnStruct.Type = 'NoteOn';
        noteOnStruct.Note = note;
        noteOnStruct.Velocity = velocity;
        noteOnStruct.Timestamp = currentTime;
        noteOnStruct.Channel = 1;
        noteOnStruct.RawBytes = [144, note, velocity, 0, 0, 0, 0, 0];
        noteOffStruct = struct();
        noteOffStruct.Type = 'NoteOff';
        noteOffStruct.Note = note;
        noteOffStruct.Velocity = 0;
        noteOffStruct.Timestamp = currentTime + duration*0.8;
        noteOffStruct.Channel = 1;
        noteOffStruct.RawBytes = [128, note, 0, 0, 0, 0, 0, 0];
        msgArray = [msgArray; noteOnStruct; noteOffStruct];
        currentTime = currentTime + duration;
    end
    writeMIDIFileFixed(msgArray, outputFile);
    fprintf('Predicted song written to %s\n', outputFile);
end

function notes = predictSongFromSeed(trainedNet, vocab, seedNotes, totalLength, temperature, topK)
    if nargin < 5
        temperature = 1.0;
    end
    if nargin < 6
        topK = 0;
    end
    durationBucket = 0;
    velocityBucket = 4;
    sequence_length = length(seedNotes);
    seedTokens = zeros(1, sequence_length);
    for i = 1:sequence_length
        key = sprintf('%d_%d_%d', seedNotes(i), durationBucket, velocityBucket);
        if isKey(vocab, key)
            seedTokens(i) = vocab(key);
        else
            seedTokens(i) = vocab('-3_-3_-3');
        end
    end
    generatedIdx = zeros(1, totalLength);
    generatedIdx(1:sequence_length) = seedTokens;
    for i = sequence_length+1:totalLength
        inputSeq = generatedIdx(i-sequence_length:i-1);
        inputSeq = reshape(inputSeq, [1, sequence_length, 1]);
        dlInput = dlarray(single(inputSeq), 'CTB');
        fakeSeqProbs = extractdata(forward(trainedNet, dlInput));
        probs = fakeSeqProbs(:, end, 1);
        if topK > 0
            nextIdx = sampleTopK(probs, topK, temperature);
        else
            nextIdx = sampleWithTemperature(probs, temperature);
        end
        generatedIdx(i) = nextIdx;
    end
    keysV = vocab.keys;
    valsV = cell2mat(vocab.values);
    reverseVocab = containers.Map(valsV, keysV);
    notes = zeros(totalLength,1);
    padToken = 60;
    for i = 1:totalLength
        if isKey(reverseVocab, generatedIdx(i))
            key = reverseVocab(generatedIdx(i));
            tokens = sscanf(key, '%d_%d_%d');
            notes(i) = tokens(1);
        else
            notes(i) = padToken;
        end
    end
end

function idx = sampleWithTemperature(probs, temperature)
    if nargin < 2
        temperature = 1.0;
    end
    probs = probs .^ (1/temperature);
    probs = probs / sum(probs);
    edges = [0; cumsum(probs(:))];
    r = rand();
    idx = find(r >= edges(1:end-1) & r < edges(2:end), 1, 'first');
    if isempty(idx)
        [~, idx] = max(probs);
    end
end

function idx = sampleTopK(probs, k, temperature)
    if nargin < 3
        temperature = 1.0;
    end
    [sortedProbs, sortedIdx] = sort(probs, 'descend');
    topKIdx = sortedIdx(1:min(k, numel(probs)));
    topKProbs = probs(topKIdx);
    topKProbs = topKProbs .^ (1/temperature);
    topKProbs = topKProbs / sum(topKProbs);
    edges = [0; cumsum(topKProbs(:))];
    r = rand();
    pick = find(r >= edges(1:end-1) & r < edges(2:end), 1, 'first');
    idx = topKIdx(pick);
end

function writeMIDIFileFixed(msgArray, filename, ticksPerQuarter)
    if nargin < 3
        ticksPerQuarter = 480;
    end
    if isempty(msgArray)
        warning('No MIDI messages to write');
        return;
    end
    try
        timestamps = [msgArray.Timestamp];
        [~, sortIdx] = sort(timestamps);
        sortedMsgArray = msgArray(sortIdx);
        fid = fopen(filename, 'w');
        if fid == -1
            error('Cannot open file %s for writing', filename);
        end
        fwrite(fid, 'MThd', 'char');
        fwrite(fid, 6, 'uint32', 'b');
        fwrite(fid, 1, 'uint16', 'b');
        fwrite(fid, 1, 'uint16', 'b');
        fwrite(fid, ticksPerQuarter, 'uint16', 'b');
        fwrite(fid, 'MTrk', 'char');
        trackLengthPos = ftell(fid);
        fwrite(fid, 0, 'uint32', 'b');
        trackData = uint8([]);
        prevTicks = 0;
        tempo = 500000;
        tempoMsg = [255, 81, 3, bitshift(tempo, -16), bitshift(tempo, -8) & 255, tempo & 255];
        deltaTime = writeVariableLength(0);
        trackData = [trackData; deltaTime; tempoMsg'];
        for i = 1:length(sortedMsgArray)
            msg = sortedMsgArray(i);
            timestampSeconds = msg.Timestamp;
            ticks = round(timestampSeconds * ticksPerQuarter * (120 / 60));
            deltaTicks = max(0, ticks - prevTicks);
            prevTicks = ticks;
            deltaTime = writeVariableLength(deltaTicks);
            rawBytes = msg.RawBytes(1:3);
            trackData = [trackData; deltaTime; rawBytes'];
        end
        deltaTime = writeVariableLength(0);
        endTrack = [255, 47, 0];
        trackData = [trackData; deltaTime; endTrack'];
        fwrite(fid, trackData, 'uint8');
        trackLength = length(trackData);
        fseek(fid, trackLengthPos, 'bof');
        fwrite(fid, trackLength, 'uint32', 'b');
        fclose(fid);
        fprintf('MIDI file written to: %s\n', filename);
    catch e
        fprintf('Error writing MIDI file: %s\n', e.message);
        if fid ~= -1
            fclose(fid);
        end
    end
end

function bytes = writeVariableLength(value)
    bytes = uint8([]);
    if value == 0
        bytes = uint8(0);
        return;
    end
    temp = [];
    while value > 0
        temp = [bitand(value, 127) temp];
        value = bitshift(value, -7);
    end
    for i = 1:length(temp)-1
        bytes = [bytes; bitor(temp(i), 128)];
    end
    bytes = [bytes; temp(end)];
end



%loading trained model and vocabulary
load('trained_music_gan.mat', 'trainedNet', 'vocab');

% Define your seed notes (as MIDI note numbers)
seedNotes = [60 62 64]; %C Major

% Set the total length of the song (including the seed)
totalLength = 64;

% Set the temperature for sampling (1.0 = default, try 0.8-1.2 for variety)
temperature = 1.0;

% Set the output MIDI filename
outputFile = 'my_seed_song.mid';

% Generate, save, and play/visualize the song
predictSongToMIDI(trainedNet, vocab, seedNotes, totalLength, temperature, outputFile);