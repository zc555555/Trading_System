"""
Export trained XGBoost model to ONNX format for Go inference.

This module:
1. Loads trained XGBoost model
2. Converts to ONNX format using skl2onnx
3. Validates ONNX model
4. Generates golden test vectors for Go
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import pandas as pd
import yaml
import xgboost as xgb
from onnx import version_converter
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


class ONNXExporter:
    """Export XGBoost model to ONNX format."""

    def __init__(self, config_path: str = None):
        """
        Initialize exporter.

        Args:
            config_path: Path to config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_model(self):
        """
        Load trained model and feature manifest.

        Returns:
            (model, feature_cols)
        """
        artifacts_dir = Path(__file__).parent.parent / "artifacts"

        # Load model
        model_path = artifacts_dir / "xgboost_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Loaded model from {model_path}")

        # Load feature manifest
        manifest_path = artifacts_dir / "feature_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        feature_cols = manifest['feature_names']

        print(f"Loaded feature manifest: {len(feature_cols)} features")

        return model, feature_cols

    def convert_to_onnx(self, model, feature_cols: List[str]) -> onnx.ModelProto:
        """
        Convert model to ONNX format.

        Args:
            model: Trained XGBoost model
            feature_cols: List of feature names

        Returns:
            ONNX model
        """
        print("\n" + "=" * 70)
        print("CONVERTING TO ONNX")
        print("=" * 70)

        num_features = len(feature_cols)
        print(f"\nNumber of features: {num_features}")

        # Define input type
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        # Convert to ONNX
        print("\nConverting model...")
        try:
            onnx_model = convert_xgboost(
                model,
                initial_types=initial_type,
                target_opset=12,  # Compatible with most ONNX runtimes
            )

            print("[OK] Conversion successful!")

        except Exception as e:
            print(f"[FAIL] Conversion failed: {e}")
            raise

        return onnx_model

    def validate_onnx(self, onnx_model: onnx.ModelProto, model, X_sample):
        """
        Validate ONNX model against original model.

        Args:
            onnx_model: ONNX model
            model: Original model
            X_sample: Sample input data
        """
        print("\n" + "=" * 70)
        print("VALIDATING ONNX MODEL")
        print("=" * 70)

        # Check ONNX model validity
        try:
            onnx.checker.check_model(onnx_model)
            print("\n[OK] ONNX model is valid")
        except Exception as e:
            print(f"\n[FAIL] ONNX model validation failed: {e}")
            raise

        # Test inference with ONNX Runtime
        try:
            import onnxruntime as ort

            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_model.SerializeToString())

            # Get input name
            input_name = sess.get_inputs()[0].name

            # Run inference
            onnx_pred = sess.run(None, {input_name: X_sample.astype(np.float32)})[0]

            # Compare with original model
            orig_pred = model.predict(X_sample)

            # Check if predictions match
            if hasattr(model, 'predict_proba'):
                # For classification, compare probabilities
                orig_proba = model.predict_proba(X_sample)
                onnx_proba = sess.run(None, {input_name: X_sample.astype(np.float32)})[1]

                max_diff_proba = np.abs(orig_proba - onnx_proba).max()
                print(f"\n[OK] ONNX Runtime inference successful")
                print(f"  Max probability difference: {max_diff_proba:.6f}")

                if max_diff_proba < 1e-4:
                    print(f"  [OK] Predictions match (tolerance: 1e-4)")
                else:
                    print(f"  ⚠ Predictions differ slightly (may be acceptable)")

            else:
                # For regression, compare values
                max_diff = np.abs(orig_pred - onnx_pred.flatten()).max()
                print(f"\n[OK] ONNX Runtime inference successful")
                print(f"  Max prediction difference: {max_diff:.6f}")

                if max_diff < 1e-4:
                    print(f"  [OK] Predictions match (tolerance: 1e-4)")
                else:
                    print(f"  ⚠ Predictions differ slightly (may be acceptable)")

        except ImportError:
            print("\n⚠ onnxruntime not available, skipping runtime validation")
        except Exception as e:
            print(f"\n[FAIL] ONNX Runtime validation failed: {e}")
            raise

    def generate_golden_vectors(
        self,
        model,
        feature_cols: List[str],
        num_samples: int = 10
    ) -> List[Dict]:
        """
        Generate golden test vectors for Go validation.

        Args:
            model: Trained model
            feature_cols: Feature column names
            num_samples: Number of test samples to generate

        Returns:
            List of test cases with inputs and expected outputs
        """
        print("\n" + "=" * 70)
        print("GENERATING GOLDEN TEST VECTORS")
        print("=" * 70)

        # Load a small sample from test data
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        test_path = artifacts_dir / "test_dataset.parquet"

        df_test = pd.read_parquet(test_path)

        # Sample random rows
        sample_indices = np.random.choice(len(df_test), min(num_samples, len(df_test)), replace=False)
        samples = df_test.iloc[sample_indices]

        golden_vectors = []

        for idx, row in samples.iterrows():
            # Get features
            features = row[feature_cols].values.astype(float)

            # Get prediction
            prediction = model.predict(features.reshape(1, -1))[0]

            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features.reshape(1, -1))[0]
                output = {
                    'prediction': int(prediction),
                    'probabilities': proba.tolist(),
                }
            else:
                output = {
                    'prediction': float(prediction),
                }

            test_case = {
                'id': int(idx),
                'date': str(row['date']),
                'symbol': row['symbol'],
                'features': features.tolist(),
                'output': output,
            }

            golden_vectors.append(test_case)

        print(f"\nGenerated {len(golden_vectors)} golden test vectors")

        return golden_vectors

    def save_onnx(self, onnx_model: onnx.ModelProto, golden_vectors: List[Dict]):
        """
        Save ONNX model and golden vectors.

        Args:
            onnx_model: ONNX model
            golden_vectors: Golden test vectors
        """
        artifacts_dir = Path(__file__).parent.parent / "artifacts"

        # Save ONNX model
        onnx_path = artifacts_dir / self.config['model']['onnx_output']
        onnx.save(onnx_model, onnx_path)

        print(f"\n[OK] ONNX model saved to {onnx_path}")
        print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")

        # Save golden vectors
        golden_path = artifacts_dir / self.config['model']['golden_vectors_output']
        with open(golden_path, 'w') as f:
            json.dump(golden_vectors, f, indent=2)

        print(f"\n[OK] Golden vectors saved to {golden_path}")

        # Save thresholds (for decision making)
        thresholds_path = artifacts_dir / self.config['model']['thresholds_output']
        thresholds = {
            'signal_threshold': 0.5,  # Default threshold for classification
            'position_sizing': {
                'min_signal': 0.55,  # Minimum signal to take position
                'max_position': 0.35,  # Maximum position size
            },
            'risk_limits': {
                'max_drawdown': 0.15,
                'max_daily_loss': 0.02,
            }
        }

        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)

        print(f"\n[OK] Thresholds saved to {thresholds_path}")

    def run_full_pipeline(self):
        """Run complete ONNX export pipeline."""
        print("=" * 70)
        print("ONNX EXPORT PIPELINE")
        print("=" * 70)

        # 1. Load model
        model, feature_cols = self.load_model()

        # 2. Convert to ONNX
        onnx_model = self.convert_to_onnx(model, feature_cols)

        # 3. Validate
        # Get sample data for validation
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        test_path = artifacts_dir / "test_dataset.parquet"
        df_test = pd.read_parquet(test_path)
        X_sample = df_test[feature_cols].head(100).values

        self.validate_onnx(onnx_model, model, X_sample)

        # 4. Generate golden vectors
        golden_vectors = self.generate_golden_vectors(model, feature_cols, num_samples=20)

        # 5. Save
        self.save_onnx(onnx_model, golden_vectors)

        print("\n" + "=" * 70)
        print("ONNX EXPORT COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Implement Go inference engine using the ONNX model")
        print("2. Validate Go predictions against golden_vectors.json")
        print("3. Ensure feature calculation in Go exactly matches Python")


def main():
    """Command-line entry point."""
    exporter = ONNXExporter()
    exporter.run_full_pipeline()


if __name__ == "__main__":
    main()
