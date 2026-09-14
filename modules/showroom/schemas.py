from datetime import datetime
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from modules.showroom.geometry import polygon_valid


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StoreCreate(Strict):
    name: str = Field(min_length=1, max_length=80)
    timezone: str = "Asia/Shanghai"
    opens: str = "09:00"
    closes: str = "18:00"
    min_dwell: float = Field(default=10, ge=1, le=600)
    max_gap: float = Field(default=2, ge=0.1, le=10)
    revisit_gap: float = Field(default=15, ge=1, le=600)

    @field_validator("timezone")
    @classmethod
    def timezone_valid(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("未知时区。") from None
        return value

    @field_validator("opens", "closes")
    @classmethod
    def valid_time(cls, value):
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("营业时间格式为 HH:MM。")
        return value

    @model_validator(mode="after")
    def hours(self):
        if self.opens >= self.closes:
            raise ValueError("本版营业时段需在同一天内，结束时间晚于开始时间。")
        return self


class Camera(Strict):
    camera_id: int = Field(ge=0, le=10000)
    width: int = Field(ge=16, le=16000)
    height: int = Field(ge=16, le=16000)
    points: list[tuple[float, float, float, float]] = Field(min_length=4, max_length=32)

    @model_validator(mode="after")
    def bounds(self):
        if any(
            not (0 <= p[0] <= self.width and 0 <= p[1] <= self.height)
            for p in self.points
        ):
            raise ValueError("摄像头标定点超出图像范围。")
        return self


class Vehicle(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")
    name: str = Field(min_length=1, max_length=80)
    model: str = Field(default="", max_length=80)
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=32)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if not polygon_valid(value):
            raise ValueError("区域不能自交、重合或没有面积。")
        return value


class LayoutCreate(Strict):
    floorplan_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    width_m: float = Field(gt=0, le=2000)
    height_m: float = Field(gt=0, le=2000)
    effective_at: datetime
    cameras: list[Camera] = Field(min_length=1, max_length=128)
    vehicles: list[Vehicle] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def valid(self):
        if self.effective_at.tzinfo is None:
            raise ValueError("生效时间必须带时区。")
        if len({c.camera_id for c in self.cameras}) != len(self.cameras) or len(
            {v.id for v in self.vehicles}
        ) != len(self.vehicles):
            raise ValueError("摄像头编号或车辆编号重复。")
        points = [p[2:] for c in self.cameras for p in c.points] + [
            p for v in self.vehicles for p in v.polygon
        ]
        if any(
            not (0 <= p[0] <= self.width_m and 0 <= p[1] <= self.height_m)
            for p in points
        ):
            raise ValueError("标定或车辆区域超出平面图实际尺寸。")
        return self


class BindingCreate(Strict):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class BindingPause(Strict):
    paused: bool


class RoleSet(Strict):
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    global_id: int = Field(ge=1)
    role: str = Field(pattern=r"^(visitor|staff|unknown)$")
