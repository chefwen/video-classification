### video classification
This is a small video classification experiment on 5 classes of human activities from UCF101 dataset.

There are mainly three models. The first one is feeding one frame of the video to normal 2D CNN, which is used as a baseline model. The second one is to use consensus function (normally maxpooling) to aggregate features from multiple frame after going through the same CNN network, and then feed the aggregated feature to the following FC layers. The third one is similar to the second one, except that after got the features from the CNN we will feed the features to LSTM. The result is summarized below.

Model | Single-frame 2D CNN | Multi-frame 2D CNN | RNN
----- | ------------------- | ------------------ | ---
test acc | 84.00% | 87.62% | 86.66%    
