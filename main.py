# import pandas as pd

df = pd.read_csv('data/new_n0_sql_data.csv')
# Remove duplicate rows
df = df.drop_duplicates()
# Remove the last column
df = df.iloc[:, :-1]
df['lba'] = df['lba'].apply(lambda x: int(x, 16))

# # Convert hex timestamps to integer, then to datetime, and finally format as YYYY-MM-DD:HH:MM:SS:ms
# df['ticks'] = pd.to_datetime(df['ticks'].apply(lambda x: int(x, 16)), unit='ns')
# df['ticks'] = df['ticks'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

# # Sort by 'ticks' in chronological order
# df = df.sort_values(by='ticks').reset_index(drop=True)

# # Rename columns to match the expected names
# df = df.rename(columns={'rd|wr': 'unique_id', 'ticks': 'ds', 'lba': 'y'})

# # Save the modified DataFrame to a new CSV file
# df.to_csv('data/modified_n0_sql_data.csv', index=False)


import pandas as pd
from prophet import Prophet

df = pd.read_csv('data/modified_output.csv')

# Select only the 'ds' and 'lba' columns
new_df = df[['ds', 'lba']]

# Rename 'lba' to 'y'
new_df = new_df.rename(columns={'lba': 'y'})

# Ensure 'ds' column is in datetime format
new_df['ds'] = pd.to_datetime(new_df['ds'])

# Ensure 'y' column is numeric
new_df['y'] = pd.to_numeric(new_df['y'])

# Fit the model with the DataFrame
m = Prophet()
trained = m.fit(new_df)
print(trained)

# Create a DataFrame to hold future dates for prediction
future = m.make_future_dataframe(periods=1000)
print(future.tail())

# Make predictions for the future dates
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
fig1.show()
fig1.savefig('plot/forecast.png')