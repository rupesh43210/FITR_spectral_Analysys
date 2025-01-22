import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import logging
from datetime import datetime


class ModelTrainer:
    """Class for training and evaluating machine learning models"""

    def __init__(self, output_dir):
        """Initialize with output directory"""
        self.output_dir = output_dir

        # Define model parameters with enhanced configurations
        self.models = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                ),
                "pipeline_steps": [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95)),  # Keep 95% of variance
                    ("classifier", None),  # Will be set in create_pipeline
                ],
            },
            "RandomForest": {
                "model": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
                "pipeline_steps": [("scaler", StandardScaler()), ("classifier", None)],
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                ),
                "pipeline_steps": [("scaler", StandardScaler()), ("classifier", None)],
            },
            "SVM": {
                "model": SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
                "pipeline_steps": [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95)),
                    ("classifier", None),
                ],
            },
            "NeuralNetwork": {
                "model": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size="auto",
                    learning_rate="adaptive",
                    max_iter=1000,
                    random_state=42,
                ),
                "pipeline_steps": [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95)),
                    ("classifier", None),
                ],
            },
        }

    def train_and_evaluate(self, X, y):
        """Train and evaluate all models"""
        results = {}
        best_f1 = 0
        best_model = None

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # For each model
        for name, config in self.models.items():
            logging.info(f"Training {name}...")

            try:
                # Create and train pipeline
                pipeline = self.create_pipeline(name, config)
                pipeline.fit(X_train, y_train)

                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)

                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                }

                # Perform cross-validation
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")

                # Log results
                logging.info(f"{name} Results:")
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.4f}")
                logging.info(
                    f"CV Scores - Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}"
                )

                # Save model
                model_path = os.path.join(
                    self.output_dir, f"{name.lower()}_model.joblib"
                )
                joblib.dump(pipeline, model_path)
                logging.info(f"Saved {name} model to {model_path}")

                # Store results
                results[name] = {
                    "model": pipeline,
                    "metrics": metrics,
                    "cv_scores": cv_scores,
                    "predictions": y_pred,
                    "probabilities": y_prob,
                }

                # Update best model
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_model = name

            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue

        if best_model:
            logging.info(f"Best model: {best_model} (F1: {best_f1:.4f})")

        return results

    def create_pipeline(self, name, config):
        """Create a pipeline for the given model"""
        steps = config["pipeline_steps"].copy()

        # Set the classifier in the pipeline
        for step in steps:
            if step[0] == "classifier":
                steps[steps.index(step)] = ("classifier", config["model"])

        return Pipeline(steps)

    def predict(self, model, X):
        """Make predictions using a trained model"""
        try:
            return model.predict(X), model.predict_proba(X)
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
