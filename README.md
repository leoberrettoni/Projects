# Machine Learning Group Project: Popularity Score of Music Tracks
Leonardo Berrettoni - Captain - 759321

Daniele Fiorucci - 761461

Francesco Migliore - 758731

## 1. Introduction

The present study on which we based our group project on, focuses on the endeavours of a prominent music streaming enterprise aimed at comprehending the underlying factors that contribute to the widespread appeal of songs hosted on their platform.           

In order to delve into the characteristics of popular songs and identify discernible patterns and trends, an exploratory data analysis (EDA) has been commissioned utilising the dataset furnished by the music intelligence department (MID). 

This dataset encompasses an array of information pertaining to songs, encompassing artist details, album titles, track names, popularity ratings, duration, and a multitude of audio features including danceability, energy, and instrumentalness.

By meticulously scrutinising this dataset, valuable insights can be gleaned regarding the constituents that engender song popularity and the particular song types that are likely to resonate with distinct user groups. Such insights can subsequently enable the company to refine and enhance recommendation models, thereby elevating the accuracy and effectiveness of their recommendations. 
Ultimately, this can significantly augment user engagement and retention on the platform.

In the following section will be discussed in the first place a detailed description of the correlations and the data distribution highlighted by the Exploratory Data Analysis alongside descriptive statistics and visualisations to comprehensively address the task.

Secondly, will be elaborated on the method selection concerning the track_genre clustering and the popularity score prediction models based on the track’s audio features.

## 2. Methods

To ensure full reproducibility of our code and the related results, it is essential for everyone reading this report to be able to recreate the environment we used for this project. 

Precisely, the code exploits essential libraries for data analysis such as NumPy and pandas. For data visualisation we made use of seaborn and matplotlib, meanwhile for machine learning purposes we relied on scikit-learn library. Lastly, for an efficient handling of large datasets we chose to utilise vaex. All these libraries offer a wide range of functions and tools to perform advanced data manipulation, visualisation, and machine learning tasks in Python.

### 2.1 Exploratory Data Analysis

The EDA task begun by conducting an initial examination of the dataset.
We employed the df.head() and df.info() function to gain a preliminary understanding of the data features and to meticulously document the features’ names and their corresponding measurements.

The main takeaways gathered about the dataset are the following:

* 18 features for 114000 observations

* 3 Null values, one each for the first 3 columns

* 4 categorical columns (first three and the last one) and one boolean column ("explicit")

We then thoroughly analysed the summary statistics of the dataset utilising the df.describe() function to swiftly obtain a comprehensive overview of the central tendencies, dispersions, and shapes of the distributions pertaining to each variable, from which it’s been evinced that the features have different scale and range.

Scrutinising the dataset for potential errors by assessing the presence of Not a Number (NaN) values within the DataFrame by employing the df.isnull().sum() function, thus enabling a precise count of such occurrences, led us to discover that there are only 3 NaN values in the dataset, one for the artists feature, one for the album_name and one for track_name.

We proceeded to validate the dataset for duplicate records and gauged the number of duplicates within the dataset by employing the df.duplicated().sum() function, thereby ensuring data integrity and accuracy. Moreover by using the .nunique() fuction on the track_id feature, we found out that out of a total of 113,549 observations, only 89,740 are unique, even though in the features description part it is written that the "track_id" is unique for each song.

A key part of the EDA regards the examination of the correlation among variables performed by employing the df.corr() function to compute the Pearson Correlation matrix of the dataset's columns and visualising the resulting correlation matrix through the utilisation of the heat-map below in order to facilitate a comprehensive understanding of the interrelationships among the variables.

![Figure 1: Heatmap](images/figure1.png)

Figure 1

From the heat-map (Figure 1) we can highlight that:

* the feature ‘energy’ has a quite high correlation with the ‘loudness’ one (0.76)
* conversely, there is a significantly negative correlation among ‘acousticness’ and ‘energy’ features (-0.73)
* the remaining features are not showing any relevant or interesting correlation
* the ‘popularity’ feature does not have any significant correlation with the other features of the dataset

Another important visualisation concerned the distribution of the dataset to explore the distribution characteristics of the dataset's variables. 

This has been crucial to ascertain whether certain variables exhibit skewed, asymmetric, or broad distributions. We accomplished this by employing df.plot(kind='hist') for a histogram representation.

![Figure 2: Features' Distribution](images/figure2.png)

Figure 2

From Figure 2 we can grasp that:

* the distribution of ‘popularity’, ‘acousticness’ and ‘instrumentalness’ appears to be “zero-inflated”
* ‘danceability’ and ‘tempo’ have a bell-shaped distribution
‘mode’ is a binary feature
* ‘duration_ms’ , ‘speechiness’ and ‘liveliness’ distributions are right-skewed
* ‘popularity’ , ‘duration_ms’ , ‘loudness’ and ‘tempo’ have a different scale
* ‘time_signature’ is a discrete variable

To count the recurrent observations of track genres we used the .describe() function on the track_genre feature, and evinced that there were exactly 1000 songs for each of the 114 music genres present in the dataset.

Lastly we decided to plot the relationships among features, precisely the one among track_genre and popularity_score in order to find out the 10 track genres with the highest cumulative popularity score.


![Figure 3: Top 10 Genres by Cumulative Popularity Score](images/figure3.png)

Figure 3

Looking at Figure 3, it is quite evident how the pop-film genre is the most popular one, followed by k-pop and chill.

It is also worth to mention the absence of more traditional genres like rock, jazz, blues or even dance or house music. This could be possibly due to the fact that those top 10 genres are among the most popular and played genres on the music streaming platform from which the data comes from and therefore represent a biased vision of the top 10 genres based on the popularity of the tracks, which is just valid to represent the sample analysed and not a applicable in general.

## 2.2 Clustering

After the EDA, it’s time to address the clustering analysis task, in which we employ an algorithm to cluster songs according to their audio characteristics.

### 2.2.1 Preprocessing

Before building the model we decided to allocate a sufficient proportion of the dataset to the test set to ensure reliable evaluation of the clustering model's performance. 
Precisely, we assigned 20% of the total dataset for testing, this allocation allows for a robust assessment of the model's ability to generalise to unseen data.

Subsequently we removed the track_genre, track_name and track_id features since they are not relevant for the analysis.

We then encoded the categorical features album_name and artists using vaex Multi Hot Encoder because similarly to One Hot Encoder does not rank the features but with respect to the latter creates a lower number of columns on the train and test dataframes.

### 2.2.2 Model Building

We employed K-Means++, selecting the number of K to be equivalent to the count of distinct music genres present in the initial dataset which is 114.
Then, we juxtaposed the output of the clustering process with the track_genre label assigned to each track.

To evaluate the K-Means++ clustering model, we took the following steps:

a cluster is assigned to every song of the dataset
we considered the most assigned cluster for every music genre
we took the latter as reference point for that specific genre
we compared the clusters assigned to any song with the reference point ones to check the accuracy of the prediction

As far as we are concerned, even though the track_genre categories were highly balanced, the accuracy of this model is incredibly low.
Precisely, the model correctly predicts the music genre cluster only 1% of the times on the training set. The situation gets worse on the test set with an accuracy of 0.821%.

## 2.3 Popularity Score Prediction

The third task regards the developing of machine learning models utilising algorithms to forecast the popularity of songs relying on their audio characteristics.

### 2.3.1 Preprocessing: Dealing with Zeros

Since from the EDA we evinced that the popularity score was a zero-inflated feature, which means that it has an excess of zero valued data points, we performed a series of operations to identify and remove songs with a popularity score of zero. 

The objective was to analyse the artists with the highest number of songs having a popularity score greater than zero. We started by creating a new data-frame which contains only the songs with a popularity score of zero by filtering the original data-frame.

Next, we created another data-frame which includes songs with a popularity score greater than zero. To identify the artists who have more songs with a popularity score of zero than songs with a popularity score greater than zero, we iterated over the 'artist_names_list' obtained from the 'artist_counts' data-frame. For each artist, we retrieved their respective counts of songs with a popularity score of zero ('count_0') and songs with a popularity score greater than zero (‘count'). If the count of songs with a popularity score of zero was greater than the count of songs with a popularity score greater than zero, we added the artist's name to the 'filtered_artist_names' list.

Once we have identified the artists whose songs with a popularity score of zero outweighed their songs with a popularity score greater than zero, we dropped their corresponding rows. When verifying the updated number of songs with a popularity score of zero, we observed still a quite high number: 8358. The amount of songs with popularity score equal to zero was still too high and we decided to use another criteria to further reduce the number.

Continuing the analysis, we created a new data-frame which again contains only the songs with a popularity score of zero. We computed the count of songs per artist with a popularity score of zero, storing the results in the 'artist_names_counts_0' data-frame.
We extracted the list of artist names ('artist_names_list_0') from the 'artist_names_counts_0' data-frame. We then initialised an empty data-frame, 'result_df', to store the results of calculating the median popularity score for each artist. For each artist in the 'artist_names_list_0', we filtered the 'df_artists_bigger_0' data-frame to include only songs by that artist. We then calculated the median popularity score for the artist and appended the artist name and median score to the 'result_df' data-frame. 
Finally, we printed the 'result_df' dataframe, which contains the artists' names and their respective median popularity scores.

Additionally, we considered a specific threshold for the median score (e.g., 20) and created a new data-frame, 'result_df_to_eliminate', that includes only the artists whose median scores are greater than or equal to this threshold. We stored the names of these artists in the 'list_to_eliminate' list. To finalise the data cleaning process, we drop the rows from the 'df' data-frame where the artist's name is in the 'list_to_eliminate' and the popularity score is zero.

In this way we reduced even more the number of zero valued popularity score data (from 8358 to 5709) and made sure that the songs removed were effectively not popular.

![Figure 4: Popularity Score Distribution](images/figure4.png)

Figure 4

### 2.3.2 Preprocessing: The Final Touches

We have already seen that there are multiple observations with the same track_id, as this feature represents a unique song that may belong to different albums or be recognised with different music genres. 

For the prediction of the popularity, we want to keep only the most popular version of each song, discarding the less popular duplicates.
Also here we double check that the track_id feature has only unique observations by implementing the .nunique() function, ending up with 89740 unique track_id entries.

The required libraries for this task are Pipeline, ColumnTransformer, MinMaxScaler, and BinaryEncoder.

After dropping track_id and track_name from the dataframe because of their low level of significance for this task, we defined popularity as dependent variable (Y) and all the remaining features as the independent variables (X) and then we selected a test set size of 25% with respect to the entire dataset to cope with the lower number of observations with respect to the original dataset and to still ensure a good balance among the two sets.

We proceeded to encode the categorical variables "album_name", "artists" and "track_genre" into binary representations using the MultiHotEncoder from the vaex module and setting the fill value for missing values as O. We then fit and transformed the train data using multi_hot_encoder_prediction.fit_transform(), while only applying the transformation to the test data. The original categorical columns, "album_name," "artists," and "track_genre," are dropped from both the train and test datasets since they are not relevant for the task.

The next step involved concatenating the encoded train sets with the original train sets using pd.concat(). In this step we aimed to combine the encoded categorical features with the remaining numerical features in X_train_prediction and X_test_prediction.

The column names in X_train_prediction and X_test_prediction are converted to strings. This conversion is necessary because the ColumnTransformer expects column names in string format.

A ColumnTransformer object, ct_prediction, is created, which applies BinaryEncoder to the "explicit" column and MinMaxScaler to the columns "duration_ms," "danceability," "key," "loudness," "speechiness," "acousticness," "valence," "tempo," and "time_signature." The remainder of the columns is left unchanged ("passthrough").

This last step is done to transform the independent variables that have been considered useful and relevant into formats suitable for subsequent machine learning modeling for the Popularity Score Prediction.

### 2.3.3 Model Building

For what concerns the Prediction model, we decided to proceed as follows:

* Simple Regression - it serves as a benchmark model
* Zero-Inflated Regression - because our Y variable in zero-inflated
* Random Forest - robust to noise and overfitting
* Neural Network - good handling of non-linear relationships

To evaluate the models that we use in the regression task we will use:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R-Squared

We opted for the first metric (RMSE) because of its higher interpretability with respect to the Mean Square Error and, for what concerns the second one (MAE), we selected it because it manages in a better way the data distribution of the dependent variable compared to RMSE. Finally R-Squared because it measures how much variability our model is catching.

## 3. Experimental Design

### 3.1 Clustering

For the clustering analysis we did not perform any experiment since it was a quite straightforward task. Moreover, in order to follow the guidelines and the specific task requirements, we could not improving the clustering per se changing for example K to find a more appropriate or better performing value.

### 3.2 Populariity Score Prediction: Regression

As mentioned in the previous chapter, our first approach to the Popularity Score prediction as a regression problem, employing at first a simple linear regression.

The Linear Regression obtained the following evaluation metrics:

* Mean Squared Error: 407.26
* Root Mean Squared Error: 20.18
* Mean Absolute Error: 16.41
* R-Squared: 0.03

The results were neither surprising nor that good. So we moved on with implementing a K-NN Regressor.

K-NN evaluation metrics:

* Mean Squared Error: 295.49
* Root Mean Squared Error: 17.19
* Mean Absolute Error: 13.34
* R-Squared: 0.30

This time we found a slight positive change in the evaluation metrics, but we were still far from acceptable results.

We then proceeded to use a Random Forest Regressor, a Zero-Inflated Regressor and an XGBoost.

The results obtained from the evaluation metrics were all very similar and very bad, with an R-Squared value around 0.35

Therefore, after those disappointing results, we decided to change approach towards the task.

We shifted the vision from regression to classification.

### 3.3 Populariity Score Prediction: Classification

This new approach saw us redifining and re-thinking the popularity variable, dividing its score into 3 classes:
* low_popularity (0-33)
* medium_popularity (33-66)
* high_popularity (66-100)

The new popularity feature will hence have the "low" class that includes tracks with a popularity score lower or equal to 33. In the "medium" class we can instead find those tracks which have a popularity score ranging between 33 and 66. Finally the "high" class is populated by the tracks boasting a popularity score above 66 and up to the maximum (e.g. 100). The range of popularity score therefore is not the same since the variable "popularity" is 0-inflated and right-skewed.

This is the distribution of the three popularity score classes.

![Figure 5: Popularity Score Classes](images/figure5.png)

Figure 5

We then proceeded to use Decision Tree, K-NN, Random Forest and XGBoost methods for classification, using Accuracy, Recall and Precision as evaluation metrics.

<br>

## 4. Results

Transforming the problem from regression to classification, allowed us to get better results with a broader precision interval and an higher consistency.

As you can see from the tables below, these are the classification metrics of all the methods both on training and test set.

Training Results 

| Method | Accuracy | Recall | Precision | 
|----------|----------|----------|----------|
|  Decision Tree  |  0.79  |   0.63  |  0.73  |
|  K - Nearest Neighbours  | 0.78  |  0.60  |   0.69  |
|  Random Forest |  0.76 |  0.53 |  0.51 |
|  XGBoost |  0.86 |  0.69 |  0.89 |

<br>

Test Results 

| Method | Accuracy | Recall | Precision | 
|----------|----------|----------|----------|
|  Decision Tree  |  0.73  |   0.56  |  0.62  |
|  K - Nearest Neighbours  | 0.77  |  0.58  |   0.66  |
|  Random Forest |  0.72 |  0.51 |  0.49 |
|  XGBoost |  0.76 | 0.55 |  0.68 |

From the results on the test set can be evinced that K-NN and XGBoost are the best performing in classifying the popularity score for a track, even if just for a tiny fraction with respect to the others in terms of Accuracy, Recall and Precision.

Moreover, by using the classification approach we shifted the popularity score prediction from a broad range (0-100) to a significantly shorter one (3 classes).
This allowed us to achieve better results on the models used.

## 5. Conclusions

### 5.1 Main Takeaways

For what concerns the clustering, it was quite expectable to see a very low performance, this can be attributed mainly to the fact that there is no scientific method to prove that a song belongs exclusively to a specific music genre (for example, that Led Zeppelin tracks belong exclusively to the rock genre).

Similar to clustering, the popularity of a song is not mathematical. It's not a magical formula. There are several personal factors that influence whether an individual appreciates a song or not, such as their mood when they hear it or the time of year (who listens to Mariah Carey or Michael Bublè in July?), or even the different situations in which they listen to music. All these things are unpredictable and human factors that cannot be always taken into account when estimating the popularity of a song.



### 5.2 Future Directions

A possible future direction for this study, could be to check the non-transferability of this type of models. 

We could have cretaed two different datframes, for a couple of artists choosen from the main dataframe, and then check the perfoemnces of the models trained with dataframe 1 on the dataframe 2 and viceversa. What we will expect from this is that when trained on a dataframe the model performs well but then when transferred to antoher dataframe its performnaces will be very bad.

## 6. References

* Code from Prof. Torre's lectures
* ChatGPT code snippets