# Music Genre Classification
This model takes raw waveforms as inputs, preprocesses the audio and classifies it as 1 of 10 predefined music genres. 

# Scripts' Organization
- GTZAN: Contains two folders, one with images (spectrograms, .png) and the other with sound files (audio, .wav) of each datapoint. There are 1000 datapoints, 100 in each of the 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. Each of the .py scripts gets classification labels from the titles of these spectrogram or sound files.
- music_trainval.py: splits the entire dataset into an 80-20 train and validation split with equal proportions of labels in each split ("stratrified"). The model is trained on the train split but both validated and tested on the val split.
- music_trainvaltest.py: splits the entire dataset into an 64-16-20 train-validation-test split with equal proportions of labels in each splis. The model is trained on the train split, validated on the val split and tested on the test split.
- The choice of the above two scripts and their respective train-val(-test) splits was made to see the effect of training data size on evaluation outcomes. 
  
# Audio Preprocessing: class AudioUtil 
- If you want to work with spectrogram images directly, then use the script images.py which will take GTZAN/images_original files as input.
- Otherwise, music_trainval.py and music_trainvaltest.py take sound files from GTZAN/genres_original files as input. Using methods class AudioUtil, it averages the number of channels in the input audio (mono), resamples (22kHz), truncates all input audios to 5000ms and converts waveform to spectrogram. The CNN itself takes this spectrogram (stored as a tensor) as input in practice (although you can input .wav files to this .py script).
  
# Results: Comparing Less and More Training Data 

## More Training Data

In music_trainval.py, we have 80 inputs per genre (800 datapoints for training, from 10 genres). The model is tested on 200 inputs from the same genres. Here is the loss plot, confusion matrix and T-SNE for the testing data:

Loss: 80-20 Train-Val Split 
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/loss_plot_trainval.png
)

Confusion Matrix: 80-20 Train-Val Split 
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/confusion_matrix_music_trainval.png
)

Confusion Matrix: 80-20 Train-Val Split 
![Example](https://github.com/manishavenkat/music-genre-classification/blob/main/tsne_plot_music_trainval.png
)

