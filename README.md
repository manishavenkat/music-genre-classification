# Music Genre Classification
This model takes raw waveforms as inputs, preprocesses the audio and classifies it as 1 of 10 predefined music genres. 

# Scripts' Organization
- GTZAN: Contains two folders, one with images (spectrograms, .png) and the other with sound files (audio, .wav) of each datapoint. There are 1000 datapoints, 100 in each of the 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. Each of the .py scripts gets classification labels from the titles of these spectrogram or sound files.
- music_trainvaltest.py: splits the entire dataset into an 64-16-20 train-validation-test split with equal proportions of labels in each splis. The model is trained on the train split, validated on the val split and tested on the test split.
  
# Audio Preprocessing: class AudioUtil 
- If you want to work with spectrogram images directly, then use the script images.py which will take GTZAN/images_original files as input.
- Otherwise, music_trainval.py and music_trainvaltest.py take sound files from GTZAN/genres_original files as input. Using methods class AudioUtil, it averages the number of channels in the input audio (mono), resamples (22kHz), truncates all input audios to 5000ms and converts waveform to spectrogram. The CNN itself takes this spectrogram (stored as a tensor) as input in practice (although you can input .wav files to this .py script).
  
# Results

In music_trainvaltest.py, we have 64 inputs per genre in training (640 datapoints for training and 160 datapoints for validation, from 10 genres). The model is tested on 200 inputs from the same genres. Here is the loss plot, confusion matrix and T-SNE for the testing data:

Loss
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/loss_plot_music_trainvaltest.png
)
The model includes early stopping. Despite this, one reason for increasing validation loss (despite decreasing training loss) could be the small size of the training set (640 training datapoints) for the parameters of this neural network. Fashion MNIST, for example, which is often used with CNNs has 60K inputs for training, incomparison. 

Confusion Matrix
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/confusion_matrix_music_trainvaltest.png
)
The model is very good at recognizing metal, classical and pop but terrible at recognizing country. 

Testing Data Results Visualisation
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/tsne_plot_music_trainvaltest.png
)
The two most distinct clusters are metal (pink dots) and classical (orange dots), whereas the other genres overlap more. Listening to the audio, you will notice that metal has a lot of vocals (speech), similar to pop and hiphop, whereas classical and jazz (to a large extent, but not all jazz) have little to no vocals (speech signals). My guess is that the the model is discriminationg primarily based on the extent of speech signals in the audio and the T-SNE is illustrating this feature. Classical and Jazz tend to gravitate towards the right-side of the plot (non-speech-prominent end?) whereas metal, pop and hiphop are more on the left-side of the plot (speech-prominent end?). 
