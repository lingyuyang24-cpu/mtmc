"""接口输入：浏览器只能调整数值参数，不能指定模型文件或执行代码。"""
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackingOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    confidence: float = Field(default=0.35, ge=0.05, le=1)
    reid_threshold: float = Field(default=0.3, gt=0, le=1)
    batch_size: int = Field(default=4, ge=1, le=32)
    min_reid_frames: int = Field(default=10, ge=1, le=64)


class StreamInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default='', max_length=80)
    url: str = Field(min_length=1, max_length=4096)

    @field_validator('url')
    @classmethod
    def validate_url(cls, value):
        value = value.strip()
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in ('rtsp', 'http', 'https') or not parsed.hostname:
                raise ValueError()
            parsed.port
            if any(ord(c) < 32 or c.isspace() for c in value):
                raise ValueError()
        except ValueError:
            raise ValueError('请输入有效的 RTSP / HTTP / HTTPS 流地址，不支持本地路径。') from None
        return value


class OnlineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    streams: list[StreamInput] = Field(min_length=1)
    options: TrackingOptions = Field(default_factory=TrackingOptions)


class OfflineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    upload_ids: list[str] = Field(min_length=1)
    options: TrackingOptions = Field(default_factory=TrackingOptions)

    @field_validator('upload_ids')
    @classmethod
    def unique_uploads(cls, value):
        if len(value) != len(set(value)):
            raise ValueError('同一上传文件不能在一个任务中重复使用。')
        return value
