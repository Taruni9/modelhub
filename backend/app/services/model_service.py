import os
import joblib
import pandas as pd

from ..config import UPLOAD_DIR, BASE_DIR


ALLOWED_EXTENSIONS = {
    ".pkl",
    ".pickle",
    ".joblib",
}


def resolve_file_path(path: str) -> str:
    """
    Resolve a model artifact file path safely across different working directories.
    """
    if not path:
        return ""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)

    # Check relative to UPLOAD_DIR
    candidate_upload = os.path.join(UPLOAD_DIR, os.path.basename(path))
    if os.path.isfile(candidate_upload):
        return os.path.abspath(candidate_upload)

    # Check relative to BASE_DIR
    candidate_base = os.path.join(str(BASE_DIR), path)
    if os.path.isfile(candidate_base):
        return os.path.abspath(candidate_base)

    # Check root workspace uploads as fallback
    workspace_upload = os.path.join(str(BASE_DIR.parent), "uploads", os.path.basename(path))
    if os.path.isfile(workspace_upload):
        return os.path.abspath(workspace_upload)

    return path


def validate_model_file(file_path: str, input_schema: dict):
    """
    Validate an uploaded machine learning model.

    Checks:
    1. Supported file extension.
    2. Model can be loaded.
    3. Model exposes a predict() method.
    4. Input schema contains a valid features list.
    5. Model can execute a test prediction.
    """

    try:
        resolved_path = resolve_file_path(file_path)

        # 1. Check file extension
        extension = os.path.splitext(resolved_path)[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            return False, "Unsupported model file format."

        # 2. Check that the file exists
        if not os.path.isfile(resolved_path):
            return False, f"Model file does not exist at {resolved_path}."

        # 3. Load the model
        loaded = joblib.load(resolved_path)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            model = loaded["pipeline"]
        else:
            model = loaded

        # 4. Check whether the model has predict()
        if not callable(getattr(model, "predict", None)):
            return (
                False,
                "Uploaded object does not provide a callable predict() method.",
            )

        # 5. Validate the input schema
        if not isinstance(input_schema, dict):
            return False, "Input schema must be a JSON object."

        if "features" not in input_schema:
            return False, "Input schema must contain 'features'."

        raw_features = input_schema["features"]

        if not isinstance(raw_features, list) or not raw_features:
            return (
                False,
                "'features' must be a non-empty list.",
            )

        feature_names = []
        for feature in raw_features:
            if isinstance(feature, str) and feature.strip():
                feature_names.append(feature.strip())
            elif (
                isinstance(feature, dict)
                and isinstance(feature.get("name"), str)
                and feature["name"].strip()
            ):
                feature_names.append(feature["name"].strip())
            else:
                return (
                    False,
                    "Every feature must have a non-empty string name.",
                )

        if len(feature_names) != len(set(feature_names)):
            return False, "Feature names must be unique."

        # 6. Create numeric test input
        test_input = [0.0] * len(feature_names)

        # 7. Test whether the model can make a prediction
        try:
            prediction = None
            try:
                prediction = model.predict([test_input])
            except Exception:
                test_df = pd.DataFrame([test_input], columns=feature_names)
                prediction = model.predict(test_df)

            if prediction is None:
                return (
                    False,
                    "Model prediction returned no result.",
                )

        except Exception as prediction_error:
            return (
                False,
                f"Test prediction failed: {str(prediction_error)}",
            )

        return (
            True,
            "Model loaded successfully and test prediction completed.",
        )

    except Exception as error:
        return False, f"Model validation failed: {str(error)}"


def save_uploaded_file(contents: bytes, filename: str):
    """
    Save an uploaded model file to storage (local and optionally S3).
    """
    from .storage_service import save_artifact
    return save_artifact(contents, filename)
