import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,log_loss  
import matplotlib.pyplot as plt
import dask.dataframe as dd



# Number of rounds
num_rounds = 10

csv_file = 'first_200_rows.csv'  
df = pd.read_csv(csv_file)

df['ticks'] = df['ticks'].apply(lambda x: int(x, 16))
df['lba'] = df['lba'].apply(lambda x: int(x, 16))
df['len_blocks'] = df['len_blocks'].apply(lambda x: int(x, 16))

label_encoder = LabelEncoder()
df['rd|wr'] = label_encoder.fit_transform(df['rd|wr'])  


X = df[['ticks', 'rd|wr', 'lba', 'len_blocks']] 
y = df['rd|wr']  

# Lists to store accuracy and loss for each round
accuracies = []
losses = []


for round_num in range(num_rounds):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #calculate loss
    y_pred_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
    losses.append(loss)

    print(f'Round {round_num+1}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}')

# Plot accuracy and loss across rounds
rounds = list(range(1, num_rounds + 1))
plt.figure(figsize=(10, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(rounds, accuracies, marker='o', color='b')
plt.title('Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(rounds, losses, marker='o', color='r')
plt.title('Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss (Log Loss)')

plt.tight_layout()
plt.show()


#-------------------------------
# Prediction for new data from "new_n0_sql_data.csv"
#-------------------------------

# Load new data for prediction
new_csv_file = 'new_n0_sql_data.csv'
# new_df = pd.read_csv(new_csv_file)
new_ddf = dd.read_csv(new_csv_file)

new_df = new_ddf.compute()


# Preprocess the new data (convert hex to int)
new_df['ticks'] = new_df['ticks'].apply(lambda x: int(x, 16))
new_df['lba'] = new_df['lba'].apply(lambda x: int(x, 16))
new_df['len_blocks'] = new_df['len_blocks'].apply(lambda x: int(x, 16))

# Encode the 'rd|wr' column for new data using the same encoder
new_df['rd|wr'] = label_encoder.transform(new_df['rd|wr'])

# Select features for prediction
X_new = new_df[['ticks', 'rd|wr', 'lba', 'len_blocks']]

# Predict using the trained model
new_predictions = model.predict(X_new)

# Decode the predictions (transform back to original labels)
new_df['prediction'] = label_encoder.inverse_transform(new_predictions)

# Print the predictions
# Save the predictions to a CSV file named "predicted_data.csv"
new_df[['ticks', 'rd|wr', 'lba', 'len_blocks', 'prediction']].to_csv('predicted_data.csv', index=False)

print(new_df[['ticks', 'rd|wr', 'lba', 'len_blocks', 'prediction']])
