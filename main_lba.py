import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss,mean_squared_error  
import matplotlib.pyplot as plt
import dask.dataframe as dd

# Number of rounds
num_rounds = 10

csv_file = 'first_200_rows.csv'  
df = pd.read_csv(csv_file)

df['orignal_lba'] = df['lba']
# Preprocess the data (convert hex to int)
df['ticks'] = df['ticks'].apply(lambda x: int(x, 16))
df['lba'] = df['lba'].apply(lambda x: int(x, 16))
df['len_blocks'] = df['len_blocks'].apply(lambda x: int(x, 16))

# Encode the 'rd|wr' column (as it will be used as a feature)
label_encoder = LabelEncoder()
df['rd|wr'] = label_encoder.fit_transform(df['rd|wr'])

# Define features (X) and target (y)
X = df[['ticks', 'rd|wr', 'len_blocks']]  # 'rd|wr' is now a feature
y = df['lba']  # Target is now 'lba'

# Lists to store accuracy and loss for each round
accuracies = []
losses = []
mse_values = []

for round_num in range(num_rounds):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

    print(f'Round {round_num+1}: Accuracy = {accuracy:.4f}, Loss = {mse:.4f}')

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
plt.plot(rounds, mse_values, marker='o', color='r')
plt.title('Loss per Round')
plt.xlabel('Round')
plt.ylabel('Mean Squared Error')

plt.tight_layout()
plt.show()

# -------------------------------
# Prediction for new data from "new_n0_sql_data.csv"
# -------------------------------

# Load new data for prediction
new_csv_file = 'new_n0_sql_data.csv'
new_ddf = dd.read_csv(new_csv_file)

# Convert to pandas dataframe
new_df = new_ddf.compute()

# Preprocess the new data (convert hex to int)
new_df['ticks'] = new_df['ticks'].apply(lambda x: int(x, 16))
new_df['lba'] = new_df['lba'].apply(lambda x: int(x, 16))
new_df['len_blocks'] = new_df['len_blocks'].apply(lambda x: int(x, 16))

# Encode 'rd|wr' column for the new data using the same encoder
new_df['rd|wr'] = label_encoder.transform(new_df['rd|wr'])

# Select features for prediction
X_new = new_df[['ticks', 'rd|wr', 'len_blocks']]

# Predict using the trained model (now predicting 'lba')
lba_predictions = model.predict(X_new)

# Convert predicted lba values to hexadecimal
lba_predictions_hex = [hex(int(pred)) for pred in lba_predictions]
new_df['orignal_lba'] = df['orignal_lba']

# Add predictions to the dataframe (hex values)
new_df['predicted_lba'] = lba_predictions_hex

# Save the predictions to a CSV file
new_df[['ticks', 'rd|wr', 'len_blocks', 'orignal_lba','predicted_lba']].to_csv('predicted_lba_data.csv', index=False)

# Print the predictions
print(new_df[['ticks', 'rd|wr', 'len_blocks', 'predicted_lba']])