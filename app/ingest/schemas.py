from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl


class WebIngestRequest(BaseModel):
    urls: list[HttpUrl]


class WebIngestFetchSuccess(BaseModel):
    status: Literal["ok"] = "ok"
    url: HttpUrl
    content_type: str
    body: str
    byte_size: int


class WebIngestFetchFailure(BaseModel):
    status: Literal["error"] = "error"
    url: HttpUrl
    reason: str


class WebIngestResponse(BaseModel):
    results: list[
        Annotated[
            WebIngestFetchSuccess | WebIngestFetchFailure, Field(discriminator="status")
        ]
    ]


class UploadIngestSuccess(BaseModel):
    status: Literal["ok"] = "ok"
    name: str
    content_type: str
    byte_size: int


class UploadIngestFailure(BaseModel):
    status: Literal["error"] = "error"
    name: str
    reason: str


class UploadIngestResponse(BaseModel):
    results: list[
        Annotated[
            UploadIngestSuccess | UploadIngestFailure, Field(discriminator="status")
        ]
    ]
