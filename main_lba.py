import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt

# Number of rounds
num_rounds = 50

# Load CSV file
csv_file = 'first_2000_rows.csv'  
df = pd.read_csv(csv_file)

# Preprocess the data (convert hex to int)
df['ticks'] = df['ticks'].apply(lambda x: int(x, 16))
df['lba'] = df['lba'].apply(lambda x: int(x, 16))
df['len_blocks'] = df['len_blocks'].apply(lambda x: int(x, 16))

# Encode the 'rd|wr' column (as it will be used as a feature)
label_encoder = LabelEncoder()
df['rd|wr'] = label_encoder.fit_transform(df['rd|wr'])

df['ticks'] = df['ticks'].apply(lambda x: np.log1p(x))
df['lba'] = df['lba'].apply(lambda x: np.log1p(x))


# Define features (X) and target (y)
X = df[['ticks', 'rd|wr','len_blocks']]  # 'rd|wr' is now a feature
y = df['lba']  # Target is 'lba'

# Lists to store R^2 and MSE values for each round
r2_scores = []
mse_values = []

for round_num in range(num_rounds):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    # Initialize RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f'Round {round_num+1}, Loss(mse) = {mse:.4f}, Accuracy(r2) = {r2:.4f}')

# Plot R^2 and MSE across rounds
rounds = list(range(1, num_rounds + 1))
plt.figure(figsize=(10, 5))

# Plot R^2 score
plt.subplot(1, 2, 1)
plt.plot(rounds, r2_scores, marker='o', color='b')
plt.title('Accuracy(r2) per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy(r2)')
# plt.savefig('R^2 per Round')

# Plot MSE
plt.subplot(1, 2, 2)
plt.plot(rounds, mse_values, marker='o', color='r')
plt.title('Loss(mse) per Round')
plt.xlabel('Round')
plt.ylabel('Loss(mse)')
plt.savefig('Loss(mse) per Round')

# Plot histogram for 'lba'
plt.figure(figsize=(10, 6))
plt.hist(df['lba'], bins=100, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of lba')
plt.xlabel('lba')
plt.ylabel('Frequency')
plt.yscale('log')  # Use log scale if values are highly skewed to see small frequencies better
plt.show()
plt.savefig('Distribution of lba')


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
new_df['orignal_lba'] = new_df['lba']
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

# Add predictions to the dataframe (hex values)
new_df['predicted_lba'] = lba_predictions_hex

# Save the predictions to a CSV file
new_df[['ticks', 'rd|wr', 'len_blocks', 'orignal_lba','predicted_lba']].to_csv('predicted_lba_data.csv', index=False)

# Print the predictions
print(new_df[['ticks', 'rd|wr', 'len_blocks', 'orignal_lba','predicted_lba']])