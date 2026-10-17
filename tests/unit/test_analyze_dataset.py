import numpy as np
import pytest

from cv.analyze_dataset import background_brightness, object_angle


def synthetic_screw(angle_deg: float, size: int = 200) -> np.ndarray:
    """Dark 'screw' on a light background: a shaft with a wide head at the angle's end."""
    img = np.full((size, size), 200, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2
    t = np.deg2rad(angle_deg)
    ux, uy = np.cos(t), -np.sin(t)  # image y axis points down; angle is counter-clockwise from +x
    along = (xx - c) * ux + (yy - c) * uy
    across = np.abs(-(xx - c) * uy + (yy - c) * ux)
    shaft = (np.abs(along) < 70) & (across < 6)
    head = (along > 55) & (along < 70) & (across < 18)
    img[shaft | head] = 60
    return img


@pytest.mark.parametrize("angle", [0, 30, 90, 135, 180, 250, 300])
def test_object_angle_recovers_direction_of_head(angle):
    measured = object_angle(synthetic_screw(angle))
    diff = (measured - angle + 180) % 360 - 180
    assert abs(diff) < 5, measured


def test_background_brightness_ignores_object():
    img = synthetic_screw(45)
    assert background_brightness(img) == pytest.approx(200, abs=1)
