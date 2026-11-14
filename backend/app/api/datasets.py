
import io
import json
import os
import uuid

import joblib
import numpy as np
import pandas as pd

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..database import get_db
from ..models import ModelArtifact
from ..services.model_service import save_uploaded_file


router = APIRouter(
    prefix="/api/datasets",
    tags=["Dataset Training"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024


def read_csv_upload(dataset_file: UploadFile):
    filename = os.path.basename(dataset_file.filename or "")

    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")

    contents = dataset_file.file.read()

    if not contents:
        raise HTTPException(400, "CSV file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "CSV must be 10 MB or smaller.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, "Could not read the CSV file.") from exc

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            400, "CSV must contain rows and at least two columns."
        )

    if not df.columns.is_unique:
        raise HTTPException(400, "CSV column names must be unique.")

    # Normalize blank cells and common missing-value representations.
    df = df.replace(r"^\s*$", np.nan, regex=True)

    return filename, df


def detect_column_type(series):
    non_null = series.dropna()

    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Numeric-looking strings should be detected as numeric.
    converted = pd.to_numeric(non_null, errors="coerce")

    if converted.notna().all():
        return "numeric"

    return "categorical"


def suggest_task(series):
    kind = detect_column_type(series)
    unique_count = series.nunique(dropna=True)

    if kind == "empty" or unique_count < 2:
        return None

    # Numeric targets with few distinct values may be class labels.
    if kind == "numeric":
        if (
            pd.api.types.is_integer_dtype(series)
            and unique_count <= min(20, max(2, int(len(series) * 0.1)))
        ):
            return "classification"

        return "regression"

    return "classification"


def inspect_dataframe(df):
    columns = []

    for name in df.columns:
        series = df[name]
        kind = detect_column_type(series)

        columns.append({
            "name": str(name),
            "data_type": kind,
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "suggested_target": False,
        })

    # Suggest a target when a plausible column can be identified.
    candidates = [
        col for col in columns
        if col["data_type"] != "empty"
        and col["unique_values"] >= 2
        and col["missing_values"] < len(df)
    ]

    target_hint = None

    if candidates:
        target_words = (
            "target", "label", "class", "outcome", "result",
            "price", "sales", "prediction", "category", "status",
        )

        named_candidates = [
            col for col in candidates
            if any(
                word in col["name"].lower()
                for word in target_words
            )
        ]

        chosen = named_candidates[-1] if named_candidates else candidates[-1]
        target_hint = chosen["name"]
        chosen["suggested_target"] = True

    task_hint = None

    if target_hint is not None:
        task_hint = suggest_task(df[target_hint])

    return {
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "suggested_target": target_hint,
        "suggested_task": task_hint,
    }


@router.post("/inspect")
def inspect_csv(
    dataset_file: UploadFile = File(...),
):
    """Inspect CSV columns before the user starts training."""
    filename, df = read_csv_upload(dataset_file)

    return {
        "success": True,
        "filename": filename,
        **inspect_dataframe(df),
    }


@router.post("/train")
def upload_csv_and_train(
    dataset_file: UploadFile = File(...),
    model_name: str = Form(...),
    target_column: str = Form(...),
    task: str = Form(...),
    db: Session = Depends(get_db),
):
    if task not in {"classification", "regression"}:
        raise HTTPException(
            400, "Task must be classification or regression."
        )

    filename, df = read_csv_upload(dataset_file)

    if target_column not in df.columns:
        raise HTTPException(
            400,
            f"Target '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}",
        )

    df = df.dropna(subset=[target_column]).copy()

    if len(df) < 5:
        raise HTTPException(
            400,
            "At least five usable rows are required for a basic "
            "train/test evaluation.",
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Remove unusable columns.
    X = X.dropna(axis=1, how="all")

    # Detect column types and retain numeric and categorical features.
    numeric_columns = [
        col for col in X.columns
        if detect_column_type(X[col]) == "numeric"
    ]

    categorical_columns = [
        col for col in X.columns
        if detect_column_type(X[col]) == "categorical"
    ]

    X = X[numeric_columns + categorical_columns]

    if X.empty:
        raise HTTPException(
            400, "No usable numeric or categorical input features found."
        )

    # Convert numeric-looking columns to numeric values.
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Convert categorical values to consistent strings.
    for col in categorical_columns:
        X[col] = X[col].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    # Remove rows with no usable features.
    usable_rows = X.notna().any(axis=1)
    X = X.loc[usable_rows]
    y = y.loc[usable_rows]

    if len(X) < 5 or y.nunique() < 2:
        raise HTTPException(
            400, "Insufficient usable rows or target classes."
        )

    label_encoder = None

    if task == "classification":
        if y.nunique() < 2:
            raise HTTPException(400, "Classification needs two classes.")

        # Encode target labels so classification works with mixed labels.
        label_encoder = LabelEncoder()
        y = pd.Series(
            label_encoder.fit_transform(y.astype(str)),
            index=y.index,
        )

        class_counts = y.value_counts()

        # Stratify only when every class can appear in both splits.
        stratify = (
            y if class_counts.min() >= 2
            and int(len(y) * 0.2) >= y.nunique()
            else None
        )

    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]

        if len(y) < 5:
            raise HTTPException(
                400, "Regression needs at least five numeric target rows."
            )

        stratify = None

    if len(X) < 5:
        raise HTTPException(400, "Not enough rows after preprocessing.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise HTTPException(
            400,
            "Could not create a suitable train/test split. "
            "Check the dataset size and target distribution.",
        ) from exc

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    if task == "classification":
        estimator = LogisticRegression(max_iter=1000)

    else:
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    try:
        # Fit only on training data; evaluate on unseen test data.
        pipeline.fit(X_train, y_train)
        test_predictions = pipeline.predict(X_test)

        if task == "classification":
            accuracy = round(
                float(accuracy_score(y_test, test_predictions)), 4
            )
            f1 = round(
                float(f1_score(
                    y_test,
                    test_predictions,
                    average="weighted",
                    zero_division=0,
                )),
                4,
            )
            precision = round(
                float(precision_score(
                    y_test,
                    test_predictions,
                    average="weighted",
                    zero_division=0,
                )),
                4,
            )
            recall = round(
                float(recall_score(
                    y_test,
                    test_predictions,
                    average="weighted",
                    zero_division=0,
                )),
                4,
            )
            cm = confusion_matrix(y_test, test_predictions).tolist()
            classes_list = [
                str(c)
                for c in (
                    label_encoder.classes_.tolist()
                    if label_encoder is not None
                    else sorted(list(set(y_test)))
                )
            ]

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_weighted": f1,
                "confusion_matrix": cm,
                "classes": classes_list,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(y_test)),
            }
        else:
            mae = round(
                float(mean_absolute_error(y_test, test_predictions)), 4
            )
            mse = round(
                float(mean_squared_error(y_test, test_predictions)), 4
            )
            rmse = round(
                float(np.sqrt(mse)), 4
            )
            r2 = (
                round(float(r2_score(y_test, test_predictions)), 4)
                if len(y_test) >= 2 else None
            )

            metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(y_test)),
            }

        # Store target labels with the model so predictions can be decoded.
        bundle = {
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "task": task,
            "target_column": target_column,
            "label_classes": (
                label_encoder.classes_.tolist()
                if label_encoder is not None else None
            ),
            "metrics": metrics,
        }

        unique_filename = f"{uuid.uuid4().hex}_trained_model.joblib"
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer)

        artifact_path = save_uploaded_file(
            buffer.getvalue(), unique_filename
        )

    except Exception as exc:
        raise HTTPException(
            400, f"Training or evaluation failed: {str(exc)}"
        ) from exc

    feature_schema = {
        "features": [
            {
                "name": str(col),
                "type": (
                    "numeric" if col in numeric_columns
                    else "categorical"
                ),
            }
            for col in X.columns
        ]
    }

    artifact = ModelArtifact(
        model_name=model_name,
        description=(
            f"Trained from {filename}; task={task}; "
            f"target={target_column}."
        ),
        framework="scikit-learn",
        version="1.0",
        filename=unique_filename,
        file_path=artifact_path,
        input_schema=json.dumps(feature_schema),
        output_description=(
            f"Prediction for target '{target_column}' ({task})."
        ),
        validation_status="VALID",
        validation_message=(
            "Training completed and held-out evaluation succeeded. "
            f"Metrics: {json.dumps(metrics)}"
        ),
    )

    try:
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
    except Exception as exc:
        db.rollback()
        try:
            os.remove(artifact_path)
        except OSError:
            pass
        raise HTTPException(
            500, "Model trained, but registration failed."
        ) from exc

    return {
        "success": True,
        "message": "Model trained, evaluated, and registered.",
        "model": {
            "id": artifact.id,
            "model_name": artifact.model_name,
            "version": artifact.version,
            "task": task,
            "target_column": target_column,
            "features": feature_schema["features"],
            "rows_used": int(len(X)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "metrics": metrics,
            "validation_status": artifact.validation_status,
        },
    }
