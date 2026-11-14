import os
import io
import logging

from ..config import (
    UPLOAD_DIR,
    STORAGE_BACKEND,
    AWS_S3_BUCKET_NAME,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)
from .model_service import resolve_file_path

logger = logging.getLogger(__name__)


def get_s3_client():
    """
    Returns an initialized boto3 S3 client if S3 is configured, else None.
    Uses IAM role credentials automatically when running on EC2, or explicit keys if provided.
    """
    if STORAGE_BACKEND != "s3" or not AWS_S3_BUCKET_NAME:
        return None

    try:
        import boto3

        client_kwargs = {}
        if AWS_REGION:
            client_kwargs["region_name"] = AWS_REGION

        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY

        return boto3.client("s3", **client_kwargs)
    except Exception as exc:
        logger.warning(
            "Could not initialize S3 client: %s. Using local file storage.",
            exc,
        )
        return None


def save_artifact(contents: bytes, filename: str) -> str:
    """
    Save an artifact locally and, if configured, upload a copy to Amazon S3.
    Returns the absolute path to the local artifact file.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    local_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))

    with open(local_path, "wb") as f:
        f.write(contents)

    s3_client = get_s3_client()
    if s3_client and AWS_S3_BUCKET_NAME:
        try:
            s3_key = f"models/{filename}"
            s3_client.upload_fileobj(
                io.BytesIO(contents),
                AWS_S3_BUCKET_NAME,
                s3_key,
            )
            logger.info(
                "Uploaded model artifact to s3://%s/%s",
                AWS_S3_BUCKET_NAME,
                s3_key,
            )
        except Exception as exc:
            logger.error(
                "Failed to upload artifact to S3: %s. Preserving local copy.",
                exc,
            )

    return local_path


def load_artifact_file(file_path: str) -> str:
    """
    Ensure the artifact is available locally for joblib loading.
    If the file is absent locally but exists in S3, downloads it to the local cache.
    """
    resolved = resolve_file_path(file_path)
    if os.path.isfile(resolved):
        return resolved

    filename = os.path.basename(file_path)
    s3_client = get_s3_client()
    if s3_client and AWS_S3_BUCKET_NAME:
        try:
            s3_key = f"models/{filename}"
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            local_target = os.path.abspath(os.path.join(UPLOAD_DIR, filename))
            s3_client.download_file(AWS_S3_BUCKET_NAME, s3_key, local_target)
            logger.info(
                "Downloaded artifact from s3://%s/%s to %s",
                AWS_S3_BUCKET_NAME,
                s3_key,
                local_target,
            )
            return local_target
        except Exception as exc:
            logger.error(
                "Could not download model artifact from S3: %s",
                exc,
            )

    return resolved
