from pydantic import BaseModel
from typing import Any, Dict


class ModelResponse(BaseModel):
    id: int
    model_name: str
    description: str | None
    framework: str
    version: str
    filename: str
    input_schema: Dict[str, Any]
    output_description: str
    validation_status: str
    validation_message: str | None
    created_at: str


class ModelListResponse(BaseModel):
    id: int
    model_name: str
    framework: str
    version: str
    validation_status: str
    created_at: str