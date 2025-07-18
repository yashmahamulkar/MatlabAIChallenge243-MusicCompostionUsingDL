% MIDI File Reader Functions

function msgArray = parseMIDIFile(filePath)
    readme = fopen(filePath, 'r');
    if readme == -1
        error('Cannot open file: %s', filePath);
    end
    [readOut, byteCount] = fread(readme);
    fclose(readme);
    ticksPerQNote = polyval(readOut(13:14), 256);
    chunkIndex = 14;
    ts = 0;              
    BPM = 120;           
    msgArray = [];
    while chunkIndex < byteCount
        chunkLength = polyval(readOut(chunkIndex+(5:8)), 256) + 8;
        ptr = 8 + chunkIndex;      
        statusByte = -1;             
        while ptr < chunkIndex + chunkLength
            [deltaTime, deltaLen] = findVariableLength(ptr, readOut);  
            ptr = ptr + deltaLen;
            [statusByte, messageLen, message] = interpretMessage(statusByte, ptr, readOut);
            [ts, msg] = createMessage(message, ts, deltaTime, ticksPerQNote, BPM);
            if ~isempty(msg) && msg.RawBytes(1) ~= 0
                msgArray = [msgArray; msg];
            end
            ptr = ptr + messageLen;
        end
        chunkIndex = chunkIndex + chunkLength;
    end
end

function [valueOut, byteLength] = findVariableLength(lengthIndex, readOut)
    byteStream = zeros(4, 1);
    for i = 1:4
        valCheck = readOut(lengthIndex + i);
        byteStream(i) = bitand(valCheck, 127);
        if ~bitand(valCheck, uint32(128))
            break
        end
    end
    valueOut = polyval(byteStream(1:i), 128);
    byteLength = i;
end

function [tsOut, msgOut] = createMessage(messageIn, tsIn, deltaTimeIn, ticksPerQNoteIn, bpmIn)
    if messageIn == -1
        tsOut = tsIn;
        msgOut = [];
        return
    end
    messageLength = length(messageIn);
    zeroAppend = zeros(8 - messageLength, 1);
    bytesIn = transpose([messageIn; zeroAppend]);
    d = double(deltaTimeIn);
    t = double(ticksPerQNoteIn);
    msPerQNote = 6e7 / bpmIn;
    timeAdd = d * (msPerQNote / t) / 1e6;
    tsOut = tsIn + timeAdd;
    try
        midiStruct = struct('RawBytes', bytesIn, 'Timestamp', tsOut);
        msgOut = midimsg.fromStruct(midiStruct);
    catch
        msgOut = [];
    end
end

function [statusOut, lenOut, message] = interpretMessage(statusIn, eventIn, readOut)
    introValue = readOut(eventIn + 1);
    if isStatusByte(introValue)
        statusOut = introValue;
        running = false;
    else
        statusOut = statusIn;
        running = true;
    end
    switch statusOut
        case 255
            [eventLength, lengthLen] = findVariableLength(eventIn + 2, readOut);
            lenOut = 2 + lengthLen + eventLength;
            message = -1;
        case 240
            [eventLength, lengthLen] = findVariableLength(eventIn + 1, readOut);
            lenOut = 1 + lengthLen + eventLength;
            message = -1;
        case 247
            [eventLength, lengthLen] = findVariableLength(eventIn + 1, readOut);
            lenOut = 1 + lengthLen + eventLength;
            message = -1;
        otherwise
            eventLength = msgnbytes(statusOut);
            if running
                lenOut = eventLength - 1;
                message = uint8([statusOut; readOut(eventIn + (1:lenOut))]);
            else
                lenOut = eventLength;
                message = uint8(readOut(eventIn + (1:lenOut)));
            end
    end
end

function n = msgnbytes(statusByte)
    if statusByte <= 191
        n = 3;
    elseif statusByte <= 223
        n = 2;
    elseif statusByte <= 239
        n = 3;
    elseif statusByte == 240
        n = 1;
    elseif statusByte == 241
        n = 2;
    elseif statusByte == 242
        n = 3;
    elseif statusByte <= 243
        n = 2;
    else
        n = 1;
    end
end

function yes = isStatusByte(b)
    yes = b > 127;
end

function [sequences, vocab, vocab_size] = processMIDIForML(msgArray, sequence_length)
    if nargin < 2
        sequence_length = 32;
    end
    noteEvents = extractNoteEvents(msgArray);
    sequences = createNoteSequences(noteEvents);
    [vocab, vocab_size] = buildVocabulary(sequences);
    sequences = sequencesToIndices(sequences, vocab);
end

function noteEvents = extractNoteEvents(msgArray)
    noteEvents = [];
    for i = 1:length(msgArray)
        msg = msgArray(i);
        msgType = msg.Type;
        if strcmp(msgType, 'NoteOn') && msg.Velocity > 0
            event = struct();
            event.type = 'NoteOn';
            event.note = msg.Note;
            event.velocity = msg.Velocity;
            event.timestamp = msg.Timestamp;
            event.channel = msg.Channel;
            noteEvents = [noteEvents; event];
        elseif strcmp(msgType, 'NoteOff') || (strcmp(msgType, 'NoteOn') && msg.Velocity == 0)
            event = struct();
            event.type = 'NoteOff';
            event.note = msg.Note;
            event.velocity = 0;
            event.timestamp = msg.Timestamp;
            event.channel = msg.Channel;
            noteEvents = [noteEvents; event];
        end
    end
end

function sequences = createNoteSequences(noteEvents)
    sequences = {};
    if isempty(noteEvents)
        return;
    end
    [~, sortIdx] = sort([noteEvents.timestamp]);
    noteEvents = noteEvents(sortIdx);
    currentSequence = [];
    prevTime = 0;
    for i = 1:length(noteEvents)
        event = noteEvents(i);
        if strcmp(event.type, 'NoteOn')
            timeDelta = event.timestamp - prevTime;
            durationBucket = min(floor(timeDelta * 4), 15);
            velocityBucket = min(floor(event.velocity / 16), 7);
            token = [event.note, durationBucket, velocityBucket];
            currentSequence = [currentSequence; token];
            prevTime = event.timestamp;
        end
    end
    chunkSize = 128;
    if size(currentSequence, 1) > chunkSize
        numChunks = ceil(size(currentSequence, 1) / chunkSize);
        for i = 1:numChunks
            startIdx = (i-1) * chunkSize + 1;
            endIdx = min(i * chunkSize, size(currentSequence, 1));
            sequences{i} = currentSequence(startIdx:endIdx, :);
        end
    else
        sequences{1} = currentSequence;
    end
end

function [vocab, vocab_size] = buildVocabulary(sequences)
    allTokens = [];
    for i = 1:length(sequences)
        seq = sequences{i};
        for j = 1:size(seq, 1)
            token = seq(j, :);
            allTokens = [allTokens; token];
        end
    end
    if isempty(allTokens)
        vocab = containers.Map();
        vocab_size = 0;
        return;
    end
    [uniqueTokens, ~, ~] = unique(allTokens, 'rows');
    specialTokens = [
        [-1, -1, -1];
        [-2, -2, -2];
        [-3, -3, -3];
    ];
    allUniqueTokens = [specialTokens; uniqueTokens];
    vocab = containers.Map();
    for i = 1:size(allUniqueTokens, 1)
        token = allUniqueTokens(i, :);
        key = sprintf('%d_%d_%d', token(1), token(2), token(3));
        vocab(key) = i;
    end
    vocab_size = size(allUniqueTokens, 1);
end

function indexedSequences = sequencesToIndices(sequences, vocab)
    indexedSequences = cell(size(sequences));
    for i = 1:length(sequences)
        seq = sequences{i};
        indexedSeq = zeros(size(seq, 1), 1);
        for j = 1:size(seq, 1)
            token = seq(j, :);
            key = sprintf('%d_%d_%d', token(1), token(2), token(3));
            if isKey(vocab, key)
                indexedSeq(j) = vocab(key);
            else
                indexedSeq(j) = vocab('-3_-3_-3');
            end
        end
        indexedSequences{i} = indexedSeq;
    end
end





function [X, Y, num_classes] = createTrainingDataFixed(sequences, sequence_length)
    if nargin < 2
        sequence_length = 32;
    end
    X = {};
    Y = {};
    for i = 1:length(sequences)
        seq = sequences{i};
        if length(seq) <= sequence_length
            continue;
        end
        startToken = 1;
        endToken = 2;
        paddedSeq = [startToken; seq; endToken];
        for j = 1:(length(paddedSeq) - sequence_length)
            inputSeq = paddedSeq(j:(j + sequence_length - 1));
            target = paddedSeq(j + sequence_length);
            X{end+1} = inputSeq';
            Y{end+1} = target;
        end
    end
    for i = 1:length(Y)
        if ~isscalar(Y{i})
            fprintf('Non-scalar Y at index %d: %s\n', i, mat2str(Y{i}));
        end
    end
    Y_numeric = cell2mat(Y);
    num_classes = length(unique(Y_numeric));
    fprintf('Created %d training samples\n', length(X));
    if ~isempty(X)
        fprintf('Sample input sequence size: %s\n', mat2str(size(X{1})));
        fprintf('Sample target value: %d\n', Y{1});
        fprintf('Target value range: %d to %d\n', min([Y{:}]), max([Y{:}]));
        fprintf('Number of unique classes in Y: %d\n', num_classes);
    end
end





% ==================== GAN TRAINING AND LOSS FUNCTIONS ====================
function sampledSeq = sampleFromGeneratorOutput(fakeSeqProbs)
    [~, sampledSeq] = max(fakeSeqProbs, [], 1);
    sampledSeq = squeeze(sampledSeq);
    if ndims(sampledSeq) == 2
        sampledSeq = reshape(sampledSeq, [1, size(sampledSeq,1), size(sampledSeq,2)]);
    end
end

function [lossD, gradientsD] = modelDiscriminatorLoss(discriminatorNet, generatorNet, realSeq, noise)
    dlXReal = dlarray(single(realSeq), 'CTB');
    dOutputReal = forward(discriminatorNet, dlXReal);
    fakeSeqProbs = forward(generatorNet, noise);
    sampledFakeSeq = sampleFromGeneratorOutput(fakeSeqProbs);
    dlSampledFakeSeq = dlarray(single(sampledFakeSeq), 'CTB');
    dOutputFake = forward(discriminatorNet, dlSampledFakeSeq);
    batchSize = size(dOutputReal,1);
    realLabels = 0.9 + 0.05*randn(batchSize,1,'like',dOutputReal);
    fakeLabels = 0.1 + 0.05*randn(batchSize,1,'like',dOutputFake);
    lossD = -mean(realLabels.*log(dOutputReal+1e-8) + (1-realLabels).*log(1-dOutputReal+1e-8) + ...
                  fakeLabels.*log(1-dOutputFake+1e-8) + (1-fakeLabels).*log(dOutputFake+1e-8));
    gradientsD = dlgradient(lossD, discriminatorNet.Learnables);
end

function [lossG, gradientsG] = modelGeneratorLoss(generatorNet, discriminatorNet, noise)
    fakeSeqProbs = forward(generatorNet, noise);
    sampledFakeSeq = sampleFromGeneratorOutput(fakeSeqProbs);
    dlSampledFakeSeq = dlarray(single(sampledFakeSeq), 'CTB');
    dOutputFake = forward(discriminatorNet, dlSampledFakeSeq);
    lossG = -mean(log(dOutputFake+1e-8));
    gradientsG = dlgradient(lossG, generatorNet.Learnables);
end

function trainedNet = trainMusicGenerationModelFromDirectory(directoryPath)
    fprintf('Scanning directory %s for MIDI files...\n', directoryPath);
    midiFiles = dir(fullfile(directoryPath, '*.mid'));
    if isempty(midiFiles)
        error('No MIDI files found in directory: %s', directoryPath);
    end
    fprintf('Found %d MIDI files\n', length(midiFiles));
    allMsgArray = [];
    for i = 1:length(midiFiles)
        filePath = fullfile(directoryPath, midiFiles(i).name);
        fprintf('Processing file %d/%d: %s\n', i, length(midiFiles), midiFiles(i).name);
        try
            msgArray = parseMIDIFile(filePath);
            allMsgArray = [allMsgArray; msgArray];         
        catch e
            fprintf('Error processing file %s: %s\n', midiFiles(i).name, e.message);
            continue;
        end
    end
    if isempty(allMsgArray)
        error('No valid MIDI messages were extracted from the files');
    end
    fprintf('Total MIDI messages collected: %d\n', length(allMsgArray));
    sequence_length = 32;
    [sequences, vocab, vocab_size] = processMIDIForML(allMsgArray, sequence_length);
    fprintf('Vocabulary size: %d\n', vocab_size);
    fprintf('Number of sequences: %d\n', length(sequences));
    [X, Y, num_classes] = createTrainingDataFixed(sequences, sequence_length);
    fprintf('Training samples: %d\n', length(X));
    fprintf('Number of unique classes: %d\n', num_classes);
    Xmat = zeros(1, sequence_length, length(X));
    for i = 1:length(X)
        Xmat(1,:,i) = X{i};
    end
    Ymat = cell2mat(Y);
    % Split into training and validation sets
    valFraction = 0.15;
    numSamples = size(Xmat,3);
    idx = randperm(numSamples);
    numVal = round(valFraction * numSamples);
    valIdx = idx(1:numVal);
    trainIdx = idx(numVal+1:end);
    XmatTrain = Xmat(:,:,trainIdx);
    XmatVal = Xmat(:,:,valIdx);
    % Define networks
    generatorLayers = [
        sequenceInputLayer(1)
        lstmLayer(128, 'OutputMode', 'sequence')
        fullyConnectedLayer(num_classes)
        softmaxLayer
    ];
    generatorNet = dlnetwork(layerGraph(generatorLayers));
    discriminatorLayers = [
        sequenceInputLayer(1)
        lstmLayer(128, 'OutputMode', 'last')
        dropoutLayer(0.3)
        fullyConnectedLayer(1)
        sigmoidLayer
    ];
    discriminatorNet = dlnetwork(layerGraph(discriminatorLayers));
    numEpochs = 50;
    batchSize = 32;
    learnRate = 0.0002;
    numBatches = floor(size(XmatTrain,3)/batchSize);
    iteration = 0;
    averageGradG = [];
    averageSqGradG = [];
    averageGradD = [];
    averageSqGradD = [];
    % Initialize loss history arrays
    lossD_history = zeros(numEpochs,1);
    lossG_history = zeros(numEpochs,1);
    valG_history = zeros(numEpochs,1);
    figure;
    for epoch = 1:numEpochs
        idx = randperm(size(XmatTrain,3));
        lossD_batches = zeros(numBatches,1);
        lossG_batches = zeros(numBatches,1);
        for b = 1:numBatches
            batchIdx = idx((b-1)*batchSize+1 : b*batchSize);
            realSeq = XmatTrain(:,:,batchIdx);
            noise = randn(1, sequence_length, batchSize, 'single');
            dlNoise = dlarray(noise, 'CTB');
            [lossD, gradientsD] = dlfeval(@modelDiscriminatorLoss, discriminatorNet, generatorNet, realSeq, dlNoise);
            iteration = iteration + 1;
            [discriminatorNet.Learnables, averageGradD, averageSqGradD] = adamupdate( ...
                discriminatorNet.Learnables, gradientsD, averageGradD, averageSqGradD, iteration, learnRate, 0.5, 0.999);
            noise = randn(1, sequence_length, batchSize, 'single');
            dlNoise = dlarray(noise, 'CTB');
            [lossG, gradientsG] = dlfeval(@modelGeneratorLoss, generatorNet, discriminatorNet, dlNoise);
            iteration = iteration + 1;
            [generatorNet.Learnables, averageGradG, averageSqGradG] = adamupdate( ...
                generatorNet.Learnables, gradientsG, averageGradG, averageSqGradG, iteration, learnRate, 0.5, 0.999);
            % Store batch losses
            lossD_batches(b) = double(gather(extractdata(lossD)));
            lossG_batches(b) = double(gather(extractdata(lossG)));
        end
        % Store average epoch loss
        lossD_history(epoch) = mean(lossD_batches);
        lossG_history(epoch) = mean(lossG_batches);
        % Validation: generator loss on validation set
        valBatchSize = min(batchSize, size(XmatVal,3));
        if valBatchSize > 0
            valNoise = randn(1, sequence_length, valBatchSize, 'single');
            dlValNoise = dlarray(valNoise, 'CTB');
            fakeSeqProbs = forward(generatorNet, dlValNoise);
            sampledFakeSeq = sampleFromGeneratorOutput(fakeSeqProbs);
            dlSampledFakeSeq = dlarray(single(sampledFakeSeq), 'CTB');
            dOutputFake = forward(discriminatorNet, dlSampledFakeSeq);
            valG_loss = -mean(log(dOutputFake+1e-8));
            valG_history(epoch) = double(gather(extractdata(valG_loss)));
        else
            valG_history(epoch) = NaN;
        end
        fprintf('Epoch %d/%d: LossD=%.4f, LossG=%.4f, ValG=%.4f\n', epoch, numEpochs, lossD_history(epoch), lossG_history(epoch), valG_history(epoch));
        plot(1:epoch, lossD_history(1:epoch), '-o', 1:epoch, lossG_history(1:epoch), '-x', 1:epoch, valG_history(1:epoch), '-s');
        legend('Discriminator Loss','Generator Loss','Val Generator Loss');
        xlabel('Epoch'); ylabel('Loss');
        title('GAN Training Progress');
        grid on;
        drawnow;
    end
    trainedNet = generatorNet;
    save('trained_music_gan.mat', 'trainedNet', 'vocab', 'vocab_size', 'sequence_length');
    fprintf('Training complete! GAN Generator saved to trained_music_gan.mat\n');
end


%  Calling Training Function 
path='dataset';
trainedNet = trainMusicGenerationModelFromDirectory(path);