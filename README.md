# deep-learning-challenge

## Analysis

The purpose of this analysis is to train a neural network model to help in classifying funding applicants. The data includes more than 34,000 organizations and associated metadata. The goal is to predict the likelihood a funded organization will be successful. Three models were trained. The second and third were an attempt to optimize from the first. Data was preprocessed and then split for training. TensorFlow and keras was used for the neural network models. 

## Results

* Data Preprocessing
  * Target variable used: IS_SUCCESSFUL
  * Feature variables: APPLICATION TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
  * Variable removed: EIN, NAME
  * Binning:
     * APPLICATION TYPE value count < 200 = 'Other'
     * CLASSIFICATION value count < 1000 = 'Other'
  * Use get_dummies to convert categorical data to numerical     
  * Use StandardScaler   
  * Training on 100 epochs. Accuracy: 72.70%
  * Model saved to AlphabetSoupCharity.h5 file

* Model 2
  * Modifications:
    * Removed STATUS column and SPECIAL CONSIDERATIONS
    * APPLICATION TYPE value count < 5 = 'Other'
    * CLASSIFICATION value count < 10 = 'Other'
    * Added 4 additional hidden layers
    * Increased number of neurons
  * Training on 100 epochs. Accuracy: 73.17%
  * The target accuracy rate of 75% was not achieved
  * Model save to AlphabetSoupCharity_Optimization.h5 file

* Model 3
  * Modifications:
    * Removed ORGANIZATION TYPE
    * APPLICATION TYPE value count < 20 = 'Other'
    * CLASSIFICATION value count < 100 = 'Other'
    * Added 4 additional hidden layers
    * Increased number of neurons
  * Training on 100 epochs. Accuracy: 72.6%
  * The target accuracy rate of 75% was not achieved
  * Model save to AlphabetSoupCharity_Optimization2.h5 file
## Summary

The initial model had an accuracy of 72.70% after training. The optimized model showed a modest increase to 73.17%. Additional optimization showed a decrease to 72.6%. None of the models achieved the target rate of 75%. Further analysis of these results is needed to be able to recommend a model. Identifying less relevant outliers in the dataset would be helpful in trying to improve the model. 

## Tools Used
Python, Jupyter, Pandas, tensorflow, sklearn, train_test_split, Keras, StandardScaler
