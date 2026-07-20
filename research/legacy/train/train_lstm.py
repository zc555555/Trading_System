"""
Train LSTM Deep Learning Model for Stock Prediction.

LSTM can capture temporal dependencies that tree-based models miss.
Expected improvement: +1-3% direction accuracy (if successful)

WARNING: This will take 30-60 minutes to train!
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def create_sequences(X, y, sequence_length=20):
    """
    Create sequences for LSTM.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        sequence_length: Number of time steps to look back

    Returns:
        X_seq: (n_samples - sequence_length, sequence_length, n_features)
        y_seq: (n_samples - sequence_length,)
    """
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape):
    """
    Build LSTM model.

    Architecture:
    - LSTM layer 1: 64 units, return sequences
    - Dropout: 0.3
    - LSTM layer 2: 32 units
    - Dropout: 0.3
    - Dense layer: 16 units, ReLU
    - Dropout: 0.2
    - Output: 1 unit (regression)
    """

    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),

        # Second LSTM layer
        layers.LSTM(32),
        layers.Dropout(0.3),

        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        # Output
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_lstm():
    """Train LSTM model."""

    print("=" * 80)
    print("LSTM DEEP LEARNING MODEL TRAINING")
    print("=" * 80)
    print("\nWARNING: This will take 30-60 minutes!")
    print("Expected improvement: +1-3% direction accuracy")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    print(f"\nDataset: {df.shape}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Features: {len(feature_cols)}")

    # Sort by date and symbol (important for time series)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Standardize features (important for LSTM)
    print("\n" + "=" * 80)
    print("STANDARDIZING FEATURES")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeature scaling complete")
    print(f"  Mean: ~0, Std: ~1")

    # Create sequences
    print("\n" + "=" * 80)
    print("CREATING TIME SEQUENCES")
    print("=" * 80)

    sequence_length = 20  # Look back 20 days

    print(f"\nSequence length: {sequence_length} days")
    print("Creating sequences...")

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

    print(f"\nSequence data shapes:")
    print(f"  X_train_seq: {X_train_seq.shape}")
    print(f"  y_train_seq: {y_train_seq.shape}")
    print(f"  X_test_seq:  {X_test_seq.shape}")
    print(f"  y_test_seq:  {y_test_seq.shape}")

    # Build model
    print("\n" + "=" * 80)
    print("BUILDING LSTM MODEL")
    print("=" * 80)

    input_shape = (sequence_length, len(feature_cols))
    model = build_lstm_model(input_shape)

    print("\nModel Architecture:")
    model.summary()

    # Train model
    print("\n" + "=" * 80)
    print("TRAINING LSTM")
    print("=" * 80)

    print("\nTraining... (this will take 30-60 minutes)")

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )

    # Train
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    # Train predictions
    train_pred = model.predict(X_train_seq, verbose=0).flatten()
    train_acc = direction_accuracy(y_train_seq, train_pred)
    train_mae = mean_absolute_error(y_train_seq, train_pred)

    print(f"\nTrain Set:")
    print(f"  Direction Accuracy: {train_acc*100:.2f}%")
    print(f"  MAE: {train_mae*100:.2f}%")

    # Test predictions
    test_pred = model.predict(X_test_seq, verbose=0).flatten()
    test_acc = direction_accuracy(y_test_seq, test_pred)
    test_mae = mean_absolute_error(y_test_seq, test_pred)
    test_r2 = r2_score(y_test_seq, test_pred)

    print(f"\nTest Set:")
    print(f"  Direction Accuracy: {test_acc*100:.2f}%")
    print(f"  MAE: {test_mae*100:.2f}%")
    print(f"  R2:  {test_r2:.4f}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    baseline_acc = 0.5902  # Tree-based ensemble

    print(f"\nTree-Based Ensemble: {baseline_acc*100:.2f}%")
    print(f"LSTM:                {test_acc*100:.2f}%")
    print(f"Improvement:         {(test_acc - baseline_acc)*100:+.2f} percentage points")

    # Save model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    model_path = artifacts_dir / "lstm_model.h5"
    model.save(model_path)
    print(f"\nLSTM model saved to: {model_path}")

    # Save scaler
    scaler_path = artifacts_dir / "lstm_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save metrics
    metrics = {
        'model_type': 'LSTM',
        'sequence_length': sequence_length,
        'train_direction_accuracy': float(train_acc),
        'train_mae': float(train_mae),
        'test_direction_accuracy': float(test_acc),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'baseline_comparison': {
            'baseline_accuracy': baseline_acc,
            'lstm_accuracy': float(test_acc),
            'improvement': float(test_acc - baseline_acc)
        },
        'training_history': {
            'epochs': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
    }

    metrics_path = artifacts_dir / "lstm_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("LSTM TRAINING COMPLETE!")
    print("=" * 80)

    if test_acc >= 0.63:
        print(f"\n[SUCCESS] Achieved {test_acc*100:.2f}% (>=63%) direction accuracy!")
    elif test_acc >= 0.60:
        print(f"\n[GOOD] Achieved {test_acc*100:.2f}% (>=60%) direction accuracy!")
    elif test_acc > baseline_acc:
        print(f"\n[IMPROVEMENT] LSTM improved by {(test_acc - baseline_acc)*100:+.2f}%")
    else:
        print(f"\n[RESULT] LSTM: {test_acc*100:.2f}%, Baseline: {baseline_acc*100:.2f}%")
        print("Deep learning did not improve over tree-based models.")


if __name__ == "__main__":
    train_lstm()
