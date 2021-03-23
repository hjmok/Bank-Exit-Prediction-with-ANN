# Bank-Exit-Prediction-with-ANN

In this project, an Artificial Neural Network is modeled to estimate if customers from three different countries are likely to leave a bank using supervised learning.



## Dataset and Library
Tensorflow was used as the main module for this model.
The dataset contains information on 10000 customers, including their geography, gender, age, credit score, estimated salary, and several other useful information. The label is whether the customer has exited the bank or not. However, the label is uneven with 7963 classified as staying and 2037 as exiting. The dataset be found from the following link:

https://www.kaggle.com/santoshd3/bank-customers

## Data Preprocessing
One of the important columns is Geography, which contain three countries: Spain, Germany, and France. However, the ANN model cannot process strings. As such, this column is One Hot Encoded so that there are three separate columns for each country, and a 0 or 1 to indicate which country the user belongs to.
Similarly, gender must also be encoded since it is given as just Male or Female. After encoding, female was assigned 0 and male was assigned 1.

The first three columns were excluded from acting as input for the model, which were Row Number, Customer ID, and Surname. This is because this information is not relevent when determining if the user will leave the bank or not. As such, the columns used for training were geography, credit score, gender, age, tenure, balance, number of products with the bank, if the user has a credit card with the bank, if the user is active, and estimated salary.

## ANN Model
The hidden layers went in the order of 12, 12, and 6 neurons. In addition, a dropout rate of 5% was added to each hidden layer. The final output size was 1, using a Sigmoid activation function to predict whether the customer will stay or leave the bank.

A batch size of 16 was chosen, Adam was used as the model optimizer, and Binary Cross Entropy was used as the loss metric since this is a binary classification problem. Early stopping was implemented to stop the training if validation loss does not decrease after 40 epochs. 

## Results
The model was set to train for 300 epochs, but stopped at epoch 167 due to Early Stopping. As seen in the Classification Report, the model has an overall accuracy of 87%. However, the model has a much higher recall for Staying compared to Exiting.

The Confusion Matrix also supports this, as it shows 268 exiting customers were misclassified as staying. This may be due to how uneven the dataset labels are (7963 stay, 2037 exit). As such, to improve the model, a more even dataset can be used so that the model can train more on the properties of customers that exited.
