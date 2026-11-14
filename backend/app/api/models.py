
import json
import os
import uuid
import math
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    Query,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelArtifact, PredictionRecord, User
from .auth import get_current_user
from ..services.model_service import (
    save_uploaded_file,
    validate_model_file,
    resolve_file_path,
)
from ..services.storage_service import load_artifact_file


router = APIRouter(
    prefix="/api/models",
    tags=["Models"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class PredictionRequest(BaseModel):
    # Supports named numeric and categorical features.
    features: dict[str, object] = Field(
        ...,
        min_length=1,
        description="Feature names and values matching the registered model schema.",
    )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def parse_json(value, fallback=None):
    """Parse JSON safely, returning a fallback if parsing fails."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback if fallback is not None else value


def normalize_prediction(value):
    """Convert NumPy values into JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, list):
        return [normalize_prediction(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): normalize_prediction(item)
            for key, item in value.items()
        }

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Prediction contains a non-finite number.")
        return value

    raise ValueError(
        f"Unsupported prediction result type: {type(value).__name__}"
    )


def load_model_and_metadata(artifact):
    """
    Load either:
      1. A standard sklearn model with predict().
      2. A training bundle created by datasets.py.

    Returns:
      model, feature_names, feature_types, task, label_classes
    """
    resolved_path = load_artifact_file(artifact.file_path)
    try:
        loaded = joblib.load(resolved_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the registered model artifact from {resolved_path}.",
        ) from exc

    feature_names = None
    feature_types = {}
    task = None
    label_classes = None

    if isinstance(loaded, dict) and "pipeline" in loaded:
        # Bundle produced by the CSV training endpoint.
        model = loaded["pipeline"]
        feature_names = loaded.get("feature_names")
        task = loaded.get("task")
        label_classes = loaded.get("label_classes")

        numeric_columns = loaded.get("numeric_columns", [])
        categorical_columns = loaded.get("categorical_columns", [])

        for name in numeric_columns:
            feature_types[name] = "numeric"

        for name in categorical_columns:
            feature_types[name] = "categorical"

    else:
        # Standard uploaded sklearn model.
        model = loaded

    if not callable(getattr(model, "predict", None)):
        raise HTTPException(
            status_code=400,
            detail="The registered artifact does not provide predict().",
        )

    # Use the registered schema as the authoritative source of feature names
    # and types. CSV-trained models also retain their training feature order.
    try:
        schema = json.loads(artifact.input_schema)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The registered model has an invalid input schema.",
        ) from exc

    schema_features = schema.get("features") if isinstance(schema, dict) else None

    if not isinstance(schema_features, list) or not schema_features:
        raise HTTPException(
            status_code=500,
            detail="The registered model has no valid feature definition.",
        )

    # New CSV-trained schema:
    # {"features": [{"name": "age", "type": "numeric"}, ...]}
    #
    # Legacy schema:
    # {"features": ["age", "income", ...]}
    schema_names = []

    for item in schema_features:
        if isinstance(item, str):
            name = item
            declared_type = None
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            name = item["name"]
            declared_type = item.get("type")
        else:
            raise HTTPException(
                status_code=500,
                detail="The registered feature schema contains an invalid feature.",
            )

        schema_names.append(name)

        if declared_type in {"numeric", "categorical"}:
            feature_types[name] = declared_type

    if feature_names:
        # Keep training order but reject inconsistent registered schemas.
        if set(feature_names) != set(schema_names):
            raise HTTPException(
                status_code=500,
                detail="The model's training features do not match its registered schema.",
            )
    else:
        feature_names = schema_names

    if len(feature_names) != len(set(feature_names)):
        raise HTTPException(
            status_code=500,
            detail="The registered feature schema contains duplicate names.",
        )

    # If this is a pipeline bundle, its preprocessing metadata takes priority.
    return model, list(feature_names), feature_types, task, label_classes


def prepare_prediction_input(features, feature_names, feature_types):
    """Validate values and build a one-row DataFrame in training column order."""
    supplied_names = set(features.keys())
    expected_names = set(feature_names)

    missing = [name for name in feature_names if name not in supplied_names]
    unexpected = [name for name in features if name not in expected_names]

    if missing or unexpected:
        messages = []

        if missing:
            messages.append("Missing features: " + ", ".join(missing))

        if unexpected:
            messages.append("Unexpected features: " + ", ".join(unexpected))

        raise HTTPException(
            status_code=422,
            detail="; ".join(messages),
        )

    row = {}

    for name in feature_names:
        value = features[name]
        kind = feature_types.get(name)

        if value is None:
            # Missing values are handled by the saved sklearn imputers.
            row[name] = np.nan
            continue

        if kind == "numeric":
            if isinstance(value, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be numeric, not boolean.",
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be a number.",
                )

            if not math.isfinite(numeric_value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be finite.",
                )

            row[name] = numeric_value

        elif kind == "categorical":
            if isinstance(value, (dict, list)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be a string or a scalar value.",
                )

            row[name] = str(value)

        else:
            # Legacy schemas may not declare types.
            if isinstance(value, (dict, list)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be a scalar value.",
                )
            if isinstance(value, bool):
                row[name] = value
            elif isinstance(value, (int, float)):
                row[name] = value
            elif isinstance(value, str):
                try:
                    num = float(value)
                    row[name] = num if math.isfinite(num) else value
                except ValueError:
                    row[name] = value
            else:
                row[name] = value

    return pd.DataFrame([row], columns=feature_names)


# --------------------------------------------------
# REGISTER A MODEL
# --------------------------------------------------

@router.post("/register")
def register_model(
    model_name: str = Form(...),
    description: str = Form(""),
    framework: str = Form(...),
    version: str = Form(...),
    input_schema: str = Form(...),
    output_description: str = Form(...),
    model_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if framework.lower() != "scikit-learn":
        raise HTTPException(
            status_code=400,
            detail="Phase 1 currently supports scikit-learn only.",
        )

    extension = os.path.splitext(model_file.filename or "")[1].lower()

    if extension not in {".pkl", ".pickle", ".joblib"}:
        raise HTTPException(
            status_code=400,
            detail="Only .pkl, .pickle and .joblib files are supported.",
        )

    try:
        parsed_schema = json.loads(input_schema)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Input schema must contain valid JSON.",
        ) from exc

    if not isinstance(parsed_schema, dict):
        raise HTTPException(
            status_code=400,
            detail="Input schema must be a JSON object.",
        )

    features = parsed_schema.get("features")

    if not isinstance(features, list) or not features:
        raise HTTPException(
            status_code=400,
            detail="'features' must be a non-empty list.",
        )

    names = []

    for feature in features:
        if isinstance(feature, str) and feature.strip():
            names.append(feature)
        elif (
            isinstance(feature, dict)
            and isinstance(feature.get("name"), str)
            and feature["name"].strip()
            and feature.get("type") in {"numeric", "categorical"}
        ):
            names.append(feature["name"])
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each feature must be a non-empty name or an object "
                    "with a name and type of numeric or categorical."
                ),
            )

    if len(names) != len(set(names)):
        raise HTTPException(
            status_code=400,
            detail="Feature names must be unique.",
        )

    contents = model_file.file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded model file is empty.",
        )

    original_filename = os.path.basename(
        model_file.filename or "model.joblib"
    )
    unique_filename = f"{uuid.uuid4()}_{original_filename}"

    file_path = save_uploaded_file(contents, unique_filename)

    try:
        is_valid, validation_message = validate_model_file(
            file_path,
            parsed_schema,
        )
    except Exception as exc:
        try:
            os.remove(file_path)
        except OSError:
            pass

        raise HTTPException(
            status_code=400,
            detail="Model validation failed.",
        ) from exc

    validation_status = "VALID" if is_valid else "INVALID"

    artifact = ModelArtifact(
        user_id=current_user.id,
        user_email=current_user.email,
        model_name=model_name,
        description=description,
        framework=framework,
        version=version,
        filename=original_filename,
        file_path=file_path,
        input_schema=json.dumps(parsed_schema),
        output_description=output_description,
        validation_status=validation_status,
        validation_message=validation_message,
    )

    try:
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
    except Exception as error:
        db.rollback()
        try:
            os.remove(file_path)
        except OSError:
            pass

        raise HTTPException(
            status_code=500,
            detail="Could not save model metadata.",
        ) from error

    return {
        "success": True,
        "message": "Model registered successfully.",
        "model": {
            "id": artifact.id,
            "user_id": artifact.user_id,
            "user_email": artifact.user_email,
            "model_name": artifact.model_name,
            "version": artifact.version,
            "framework": artifact.framework,
            "validation_status": artifact.validation_status,
            "validation_message": artifact.validation_message,
        },
    }


# --------------------------------------------------
# LIST REGISTERED MODELS
# --------------------------------------------------

@router.get("/")
def get_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    models = (
        db.query(ModelArtifact)
        .order_by(ModelArtifact.created_at.desc())
        .all()
    )

    return [
        {
            "id": model.id,
            "user_id": model.user_id,
            "user_email": model.user_email or "System",
            "is_owner": model.user_id == current_user.id,
            "model_name": model.model_name,
            "framework": model.framework,
            "version": model.version,
            "validation_status": model.validation_status,
            "metrics": parse_json(
                model.validation_message.split("Metrics: ", 1)[1]
                if model.validation_message and "Metrics: " in model.validation_message
                else None,
                fallback=None,
            ),
            "created_at": model.created_at.isoformat(),
        }
        for model in models
    ]


# --------------------------------------------------
# GET PREDICTION HISTORY
# --------------------------------------------------

@router.get("/predictions/history")
def get_prediction_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of prediction records to return.",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    history = []

    for record in records:
        history.append({
            "id": record.id,
            "user_id": record.user_id,
            "user_email": record.user_email or current_user.email,
            "model_id": record.model_id,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "input_features": parse_json(record.input_features),
            "prediction": parse_json(record.prediction),
            "output_description": record.output_description,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
        })

    return {
        "success": True,
        "count": len(history),
        "history": history,
    }


# --------------------------------------------------
# GET ANALYTICS SUMMARY
# --------------------------------------------------

@router.get("/analytics/summary")
def get_analytics_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total_models = db.query(ModelArtifact).count()

    valid_models = (
        db.query(ModelArtifact)
        .filter(ModelArtifact.validation_status == "VALID")
        .count()
    )

    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .all()
    )

    total_predictions = len(records)
    successful_predictions = sum(
        1 for record in records if record.status == "SUCCESS"
    )
    failed_predictions = total_predictions - successful_predictions

    success_rate = (
        round(successful_predictions / total_predictions * 100, 2)
        if total_predictions
        else 0
    )

    prediction_counts = Counter()
    model_counts = Counter()

    for record in records:
        model_counts[record.model_name] += 1
        prediction = parse_json(record.prediction)

        model_name = record.model_name.lower()

        # Keep the existing display mapping for Iris demo models.
        if "iris" in model_name:
            iris_classes = {
                0: "Iris setosa",
                1: "Iris versicolor",
                2: "Iris virginica",
            }

            try:
                label = iris_classes.get(int(prediction), str(prediction))
            except (TypeError, ValueError):
                label = str(prediction)
        else:
            label = str(prediction)

        prediction_counts[label] += 1

    return {
        "total_models": total_models,
        "valid_models": valid_models,
        "total_predictions": total_predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "success_rate": success_rate,
        "prediction_distribution": [
            {"name": name, "count": count}
            for name, count in prediction_counts.items()
        ],
        "model_usage": [
            {"name": name, "count": count}
            for name, count in model_counts.items()
        ],
    }


# --------------------------------------------------
# GET MODEL DETAILS
# --------------------------------------------------

@router.get("/{model_id}")
def get_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    artifact = (
        db.query(ModelArtifact)
        .filter(ModelArtifact.id == model_id)
        .first()
    )

    if not artifact:
        raise HTTPException(
            status_code=404,
            detail="Model not found.",
        )

    return {
        "id": artifact.id,
        "user_id": artifact.user_id,
        "user_email": artifact.user_email or "System",
        "is_owner": artifact.user_id == current_user.id,
        "model_name": artifact.model_name,
        "description": artifact.description,
        "framework": artifact.framework,
        "version": artifact.version,
        "filename": artifact.filename,
        "input_schema": parse_json(artifact.input_schema),
        "output_description": artifact.output_description,
        "validation_status": artifact.validation_status,
        "validation_message": artifact.validation_message,
        "created_at": artifact.created_at.isoformat(),
    }


# --------------------------------------------------
# MAKE A PREDICTION AND SAVE IT TO HISTORY
# --------------------------------------------------

@router.post("/{model_id}/predict")
def predict(
    model_id: int,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    artifact = (
        db.query(ModelArtifact)
        .filter(ModelArtifact.id == model_id)
        .first()
    )

    if not artifact:
        raise HTTPException(
            status_code=404,
            detail=f"Model with ID {model_id} was not found.",
        )

    if artifact.validation_status != "VALID":
        raise HTTPException(
            status_code=400,
            detail="This model has not passed validation and cannot be used for predictions.",
        )

    resolved_path = load_artifact_file(artifact.file_path)
    if not os.path.isfile(resolved_path):
        raise HTTPException(
            status_code=404,
            detail="The model artifact file could not be found on the server.",
        )

    model, feature_names, feature_types, task, label_classes = (
        load_model_and_metadata(artifact)
    )

    dataframe = prepare_prediction_input(
        request.features,
        feature_names,
        feature_types,
    )

    try:
        # Pipeline bundles accept the original named columns, so their
        # saved imputer, scaler and encoder are reused at inference time.
        try:
            raw_prediction = model.predict(dataframe)
        except Exception:
            # Fallback for models fitted strictly on NumPy arrays without feature names
            raw_prediction = model.predict(dataframe.values)

        prediction_value = normalize_prediction(raw_prediction)

        if isinstance(prediction_value, list):
            if not prediction_value:
                raise ValueError("The model returned an empty prediction.")
            prediction_value = prediction_value[0]

        # Decode class labels for models trained by datasets.py.
        if task == "classification" and label_classes is not None:
            try:
                class_index = int(prediction_value)
                if 0 <= class_index < len(label_classes):
                    prediction_value = str(label_classes[class_index])
            except (TypeError, ValueError):
                pass
        elif task == "regression" and isinstance(prediction_value, (int, float)):
            prediction_value = round(float(prediction_value), 4)

        prediction_json = json.dumps(prediction_value, allow_nan=False)
        input_features_json = json.dumps(
            request.features,
            allow_nan=False,
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error

    record = PredictionRecord(
        user_id=current_user.id,
        user_email=current_user.email,
        model_id=artifact.id,
        model_name=artifact.model_name,
        model_version=artifact.version,
        input_features=input_features_json,
        prediction=prediction_json,
        output_description=artifact.output_description,
        status="SUCCESS",
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Prediction was generated, but its history record could not be saved.",
        ) from error

    return {
        "success": True,
        "history_id": record.id,
        "model_id": artifact.id,
        "model_name": artifact.model_name,
        "version": artifact.version,
        "input_features": request.features,
        "prediction": prediction_value,
        "output_description": artifact.output_description,
        "created_at": record.created_at.isoformat(),
    }