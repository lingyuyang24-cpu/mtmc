"""接口输入：浏览器只能调整受控的追踪参数，不能指定模型文件或执行代码。"""
from typing import Literal
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrackingOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    # Defaults mirror start_accuracy.ps1 so Web and terminal runs begin from
    # the same reviewed high-accuracy profile.
    confidence: float = Field(default=0.20, ge=0.05, le=1)
    detector_imgsz: int = Field(default=1280, ge=320, le=2048)
    batch_size: int = Field(default=2, ge=1, le=32)

    deep_sort_max_cosine_distance: float = Field(default=0.20, gt=0, le=2)
    tracker_max_age: int = Field(default=30, ge=1, le=600)
    nms_max_overlap: float = Field(default=0.35, ge=0, le=1)
    tracker_new_track_min_confidence: float = Field(default=0.25, ge=0, le=1)
    track_duplicate_containment: float = Field(default=0.85, ge=0, le=1)
    track_duplicate_max_cosine_distance: float = Field(default=0.15, ge=0, le=2)

    reid_threshold: float = Field(default=0.35, gt=0, le=1)
    global_reid_strong_threshold: float = Field(default=0.25, gt=0, le=1)
    global_reid_margin: float = Field(default=0.02, ge=0, le=1)
    global_gallery_match: Literal['topk', 'centroid', 'hybrid', 'adaptive', 'bidirectional'] = 'adaptive'
    global_candidate_threshold: float = Field(default=0.45, gt=0, le=2)
    global_borderline_confirm_frames: int = Field(default=15, ge=1, le=300)
    global_borderline_confirm_ratio: float = Field(default=0.80, gt=0, le=1)
    global_borderline_confirm_threshold: float = Field(default=0.35, gt=0, le=2)
    global_prototype_count: int = Field(default=6, ge=0, le=64)
    global_prototype_merge_threshold: float = Field(default=0.15, ge=0, le=2)

    global_feature_min_confidence: float = Field(default=0.60, ge=0, le=1)
    global_feature_min_box_height: int = Field(default=96, ge=1, le=4096)
    global_feature_max_occlusion: float = Field(default=0.20, ge=0, le=1)
    global_feature_update_max_distance: float = Field(default=0.35, gt=0, le=2)

    same_camera_reconnect: bool = True
    same_camera_reconnect_timeout: float = Field(default=1800, ge=0, le=86400)
    same_camera_reconnect_distance: float = Field(default=1.0, ge=0, le=10)
    same_camera_reconnect_reid_threshold: float = Field(default=0.50, gt=0, le=2)
    same_camera_reconnect_reid_margin: float = Field(default=0.05, ge=0, le=1)
    same_camera_reconnect_confirm_frames: int = Field(default=10, ge=1, le=300)
    same_camera_reconnect_confirm_ratio: float = Field(default=0.80, gt=0, le=1)
    same_camera_reconnect_confirm_threshold: float = Field(default=0.50, gt=0, le=2)
    same_camera_conflict_continuity: float = Field(default=5.0, ge=0, le=300)

    min_reid_frames: int = Field(default=10, ge=1, le=64)
    offline_identity_mode: Literal['streaming', 'posthoc'] = 'streaming'
    offline_workers: int = Field(default=2, ge=1, le=4, strict=True)
    offline_gallery_size: int = Field(default=32, ge=1, le=256)
    offline_gallery_match: Literal['pairwise', 'bidirectional'] = 'bidirectional'

    @model_validator(mode='after')
    def validate_related_thresholds(self):
        if self.global_reid_strong_threshold > self.reid_threshold:
            raise ValueError('强匹配阈值不能大于常规跨镜匹配阈值。')
        if self.global_borderline_confirm_threshold > self.reid_threshold:
            raise ValueError('临界确认阈值不能大于常规跨镜匹配阈值。')
        if self.global_candidate_threshold < self.reid_threshold:
            raise ValueError('候选检索上限不能小于常规跨镜匹配阈值。')
        return self


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
