## HaSpam

#### app.py:

#### Spam detection is based on text lengh and number of puncutations. 

#### The dataset is visualized to show the pattern of these features in ham and spam messages

#### Random Forest is applied and despite its reasonable test accuracy rate (almost %89), the cofusion matrix crticizes the performance of model

#### main.py:

#### Spam detection is based on message content. 

#### Text is vectorized using TFIDF technique 

#### A pipleline is employed to first vectorize then load the dat to the model 

#### Random Forest is applied and that outperforms the previous model significantly by %97 test accuracy and huge imporvements in cofusion matrix

#### Lastly, the model is tested by user made samples
