## Earthquake-Prediction
This repository is associated with various machine learning and optimization techniques used for the prediction of earthquakes. Various methods are applied taking the raw fields of the original dataset as well as certain useful features extracted from the original dataset as input to the various machine learning models.It was observed that the results improved when the features extracted from the original dataset were used as input to those machine learning models than when the raw fields of the original data were used for the same purpose.
## Description of various datasets 
#### database_original.csv
This dataset was taken from Kaggle and the url link for the same is https://www.kaggle.com/usgs/earthquake-database. All the input fields of the dataset are well described at the mentioned link above.
#### Dataset.csv
As mentioned earlier various features were extracted from the original dataset. These features are same as those mentioned in the research paper titled "NEURAL NETWORK MODELS FOR EARTHQUAKE MAGNITUDE PREDICTION USING MULTIPLE SEISMICITY INDICATORS" by Ashif Panakkat and Hojjat Adeli. All these features are extracted through programming and saved as "Dataset.csv". There are a total of eight features mentioned in the above reference.
#### ext_features.csv
Six out of the eight features were finally considered for the whole learning process. As the last two features produced a very large number of Zeroes, they were discarded. So this file contains the six of those extracted features only.
#### quake_norm.csv
This csv file contains the normalized values of all the extracted features in the ext_features.csv file.
## Neural network model used
Functional Link Artificial Neural Network(FLANN).
## Optimization techniques used
#### Heuristic optimization
1. Particle Swarm Optimization Algorithm.
2. Moth Flame Optimization Algorithm(Implementation by Seyedali Mirjalili whose details can be seen here http://www.alimirjalili.com/MFO.html).
#### Machine learning algorithms
1. Gradient descent optimization
2. Levenberg-Marquardt Backpropagation algorithm
Both of these above algorithms are implemented using the neural network toolboox. The source code for this implementatio is in the "nn_lmbp_gd.m" file.
