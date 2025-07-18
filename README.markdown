# Music Composition with Deep Learning - MathWorks AI Challenge 2025

This project is a submission for the [MathWorks AI Challenge 2025](https://www.mathworks.com/campaigns/offers/ai-challenge.html), addressing the challenge of designing and training a deep learning model to compose music. The project implements a Generative Adversarial Network (GAN) in MATLAB to generate music sequences from MIDI files, showcasing the potential of AI in creating new musical assets. The solution leverages LSTM-based neural networks to process MIDI data and generate novel compositions based on seed notes.

## Table of Contents
- [Overview](#overview)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Function Descriptions](#function-descriptions)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Example](#example)
- [Submission Details](#submission-details)
- [License](#license)

## Overview
This project develops a music generation system using a GAN trained on MIDI files. It includes a pipeline to:
- Parse MIDI files and extract note events.
- Preprocess MIDI data into sequences suitable for machine learning.
- Train a GAN model with LSTM-based generator and discriminator networks.
- Generate new MIDI songs from seed notes using the trained model.
- Save generated songs as MIDI files.

The project aligns with the MathWorks AI Challenge's focus on innovative AI solutions, demonstrating real-world applicability in generative music creation, novelty through GAN-based music generation, high-quality MATLAB code, and a deep approach to MIDI processing and sequence generation.




## Model Performance

Below is a plot showing the model's performance during training:

![Model Performance](Figure_3.png)

Epoch 50/50: LossDiscriminator=0.6753, LossGenerator=0.1106, ValGenerator=0.1186

## Installation
1. Ensure MATLAB (R2019b or later) is installed with the Deep Learning Toolbox.
2. Clone or download this repository from [GitHub](https://github.com/yashmahamulkar/MatlabAIChallenge243-MusicCompostionUsingDL.git) to your local machine.
3. Place your MIDI files in a `dataset` directory within the project folder.
4. Add all provided MATLAB functions to your MATLAB path.

## Usage
1. **Prepare MIDI Files**: Place your MIDI files (`.mid`) in the `dataset` folder.
2. **Train the Model**: Run the `trainMusicGenerationModelFromDirectory` function to process MIDI files and train the GAN model.
3. **Generate a Song**: Use the `predictSongToMIDI` function to generate a new MIDI song from seed notes.
4. **Optional Video Tutorial**: Watch our [YouTube video tutorial](https://youtu.be/J8Sh1GmUaEQ?si=TH6htUPiNlF8B_H6) for a step-by-step guide on running the project (replace with actual link if submitted).

### Example Workflow
```matlab
% Train the model
path = 'dataset';
trainedNet = trainMusicGenerationModelFromDirectory(path);

% Load trained model and vocabulary
load('trained_music_gan.mat', 'trainedNet', 'vocab');

% Define seed notes (e.g., C Major chord)
seedNotes = [60 62 64];

% Set parameters
totalLength = 64;
temperature = 1.0;
outputFile = 'my_seed_song.mid';

% Generate and save a new song
predictSongToMIDI(trainedNet, vocab, seedNotes, totalLength, temperature, outputFile);
```

The generated song will be saved as `my_seed_song.mid` in the project directory.

## Function Descriptions
- **`parseMIDIFile(filePath)`**: Parses a MIDI file to extract messages, including note events and timing.
- **`findVariableLength(lengthIndex, readOut)`**: Decodes variable-length quantities in MIDI files.
- **`createMessage(messageIn, tsIn, deltaTimeIn, ticksPerQNoteIn, bpmIn)`**: Creates a MIDI message struct with timestamp and raw bytes.
- **`interpretMessage(statusIn, eventIn, readOut)`**: Interprets MIDI message bytes based on status and running status.
- **`msgnbytes(statusByte)`**: Determines the number of bytes for a MIDI message based on its status byte.
- **`isStatusByte(b)`**: Checks if a byte is a MIDI status byte.
- **`processMIDIForML(msgArray, sequence_length)`**: Processes MIDI messages into sequences for machine learning.
- **`extractNoteEvents(msgArray)`**: Extracts note-on and note-off events from MIDI messages.
- **`createNoteSequences(noteEvents)`**: Creates sequences of note tokens with duration and velocity.
- **`buildVocabulary(sequences)`**: Builds a vocabulary of unique note tokens.
- **`sequencesToIndices(sequences, vocab)`**: Converts note sequences to indexed sequences using the vocabulary.
- **`createTrainingDataFixed(sequences, sequence_length)`**: Prepares training data for the GAN model.
- **`sampleFromGeneratorOutput(fakeSeqProbs)`**: Samples sequences from the generator's probability output.
- **`modelDiscriminatorLoss(discriminatorNet, generatorNet, realSeq, noise)`**: Computes the discriminator loss for GAN training.
- **`modelGeneratorLoss(generatorNet, discriminatorNet, noise)`**: Computes the generator loss for GAN training.
- **`trainMusicGenerationModelFromDirectory(directoryPath)`**: Trains the GAN model on MIDI files in the specified directory.
- **`predictSongToMIDI(trainedNet, vocab, seedNotes, totalLength, temperature, outputFile)`**: Generates a MIDI song from seed notes using the trained model.
- **`predictSongFromSeed(trainedNet, vocab, seedNotes, totalLength, temperature, topK)`**: Generates note indices from seed notes.
- **`sampleWithTemperature(probs, temperature)`**: Samples from a probability distribution with temperature scaling.
- **`sampleTopK(probs, k, temperature)`**: Samples from the top-k probabilities with temperature scaling.
- **`writeMIDIFileFixed(msgArray, filename, ticksPerQuarter)`**: Writes MIDI messages to a MIDI file.
- **`writeVariableLength(value)`**: Encodes a value as a variable-length quantity for MIDI files.

## File Structure
```
project_root/
│
├── dataset/                 % Directory containing input MIDI files
├── trained_music_gan.mat    % Saved trained model and vocabulary
├── my_seed_song.mid         % Generated MIDI song
├── README.md                % This file
├── LICENSE                  % MIT license file
└── *.m                      % MATLAB source files
```

## Dependencies
- MATLAB (R2019b or later)
- Deep Learning Toolbox
- MIDI files (`.mid`) for training

## Example
To train a model and generate a song:
1. Place MIDI files in the `dataset` folder.
2. Run the training script:
   ```matlab
   path = 'dataset';
   trainedNet = trainMusicGenerationModelFromDirectory(path);
   ```
3. Generate a song with seed notes:
   ```matlab
   load('trained_music_gan.mat', 'trainedNet', 'vocab');
   seedNotes = [60 62 64]; % C Major chord
   totalLength = 64;
   temperature = 1.0;
   outputFile = 'my_seed_song.mid';
   predictSongToMIDI(trainedNet, vocab, seedNotes, totalLength, temperature, outputFile);
   ```
4. The generated song will be saved as `my_seed_song.mid` and can be played using any MIDI-compatible software.

## Submission Details
This project is submitted for the MathWorks AI Challenge 2025, addressing the "Design and train a deep learning model to compose music" project from the [Artificial Intelligence technology trends list](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/megatrends/Artificial_Intelligence.md). The submission meets the following challenge requirements:
- **Real-World Applicability**: The generative music model can create new musical assets for media, gaming, or creative industries, enabling on-demand music generation.[](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/megatrends/Artificial%2520Intelligence.md)
- **Novelty**: The use of a GAN with LSTM networks for MIDI-based music generation provides a novel approach to composing music with AI.
- **Quality of Code, Models, and Documentation**: The MATLAB code is modular, well-documented, and includes a comprehensive README with usage instructions. The GAN model is optimized for music sequence generation.
- **Depth of Solution**: The project includes a full pipeline from MIDI parsing to training and generation, with advanced features like temperature-based sampling and vocabulary indexing.



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.