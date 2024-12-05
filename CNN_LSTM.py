import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Dropout
import dask.array as da
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

def r2_score(y_true, y_pred):
    """
    R-squared (coefficient of determination) metric.
    """
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # Add epsilon to avoid division by zero


def load_and_preprocess_data(file_path, request_type, sequence_length,limit=None):
    """
    Load data, filter by request type, normalize, and create sequences.
    """
    # Load data
    df = pd.read_csv(file_path, sep=' ', header=None, nrows=limit)
    df.columns = ['timestamp', 'request_type', 'LBA', 'request_size', 'access_type', 'arrival_time', 'service_time', 'idle_time']
    
    # Filter data for specific request type and required columns
    df = df[df['request_type'] == request_type]
    df = df[['timestamp', 'request_type', 'LBA']]
    
    # Normalize the 'LBA' column
    df['LBA'] = (df['LBA'] - df['LBA'].min()) / (df['LBA'].max() - df['LBA'].min())
    
    # Create sequences
    sequences = create_sequences(df['LBA'].values, sequence_length)
    
    return sequences


def create_sequences(data, sequence_length):
    """
    Create sequences for time-series input using dask for efficient processing.
    """
    num_sequences = len(data) - sequence_length
    data_dask = da.from_array(data, chunks=(num_sequences // 4,))
    sequences = da.lib.stride_tricks.sliding_window_view(data_dask, sequence_length)
    return sequences.compute()


def split_data(sequences, train_ratio=0.8):
    """
    Split the data into train and test sets.
    """
    train_size = int(len(sequences) * train_ratio)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    
    # Separate inputs (X) and outputs (y)
    X_train = train_sequences[:, :-1]
    y_train = train_sequences[:, -1]
    X_test = test_sequences[:, :-1]
    y_test = test_sequences[:, -1]
    
    # Reshape inputs for CNN-LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))  # Add channel dimension
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))
    
    return X_train, y_train, X_test, y_test


def build_model(input_shape):
    """
    Build the CNN-LSTM model.
    """
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (1, 1), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(1, 1))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Single output for regression
    
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae',r2_score])
    return model


def plot_training_history(history):
    """
    Plot the training and validation loss over epochs.
    """
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.legend()
    # plt.show()
        # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot R-squared
    plt.subplot(1, 3, 3)
    plt.plot(history.history['r2_score'], label='Train R-squared')
    plt.plot(history.history['val_r2_score'], label='Validation R-squared')
    plt.title('R-squared (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('R-squared')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Set threading configuration for CPU
    os.environ['TF_NUM_INTEROP_THREADS'] = '100'  # Adjust based on your CPU cores
    os.environ['TF_NUM_INTRAOP_THREADS'] = '100'
    tf.config.threading.set_inter_op_parallelism_threads(100)
    tf.config.threading.set_intra_op_parallelism_threads(100)

    # Parameters
    file_path = "data/msr_trace/src1_1.revised"
    request_type = 'RS'
    sequence_length = 10
    epochs = 20
    batch_size = 32
    limit = 7000000

    # Step 1: Load and preprocess data
    sequences = load_and_preprocess_data(file_path, request_type, sequence_length,limit=limit)
    X_train, y_train, X_test, y_test = split_data(sequences)

    # Step 2: Create parallelized datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Step 2: Build the model
    model = build_model(input_shape=(X_train.shape[1], 1, 1, 1))

    # # Step 3: Train the model
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     validation_data=(X_test, y_test),
    #     verbose=1
    # )
    # Step 5: Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=1
    )

    # Step 4: Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

    # Step 5: Save the model
    model.save("cnn_lstm_model_simple.h5")

    # Step 6: Plot training history
    plot_training_history(history)


if __name__ == "__main__":
    main()
