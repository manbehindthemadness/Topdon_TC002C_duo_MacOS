from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from topdon_duo.recording import VideoRecorder


@pytest.fixture
def writer(monkeypatch):
    writer = Mock()
    writer.isOpened.return_value = True
    writer.frames = []
    writer.write.side_effect = lambda image: writer.frames.append(image.copy())
    monkeypatch.setattr("topdon_duo.recording.cv2.VideoWriter", Mock(return_value=writer))
    return writer


def test_video_timing_preserves_duration_with_irregular_frame_delivery(writer, tmp_path):
    recorder = VideoRecorder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.start(tmp_path / "video.mp4", image.shape, "video", now=10.0)
    assert recorder.write(image, now=10.0) == 1
    image[:] = 50
    assert recorder.write(image, now=10.02) == 0
    image[:] = 100
    assert recorder.write(image, now=10.12) == 3
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 50, 50, 100]
    assert recorder.frames_written == 4
    assert recorder.elapsed_seconds(now=10.12) == pytest.approx(0.12)
    assert recorder.stop() == tmp_path / "video.mp4"
    assert not recorder.is_recording
    assert recorder.stop() is None
    assert recorder.write(image, now=11.0) == 0
    writer.release.assert_called_once()


@pytest.mark.parametrize("frames_per_minute", [30, 60, 120, 600, 1500])
def test_timelapse_samples_selected_frames_per_minute(writer, tmp_path, frames_per_minute):
    recorder = VideoRecorder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    interval = 60 / frames_per_minute
    recorder.start(
        tmp_path / "lapse.mp4",
        image.shape,
        "timelapse",
        frames_per_minute=frames_per_minute,
        now=10.0,
    )
    assert recorder.write(image, now=10.0) == 1
    assert recorder.write(image, now=10.0 + interval / 2) == 0
    assert recorder.write(image, now=10.0 + interval) == 1
    # After a stall, only capture the current sample, then resume the schedule.
    assert recorder.write(image, now=10.0 + interval * 5.1) == 1
    assert recorder.write(image, now=10.0 + interval * 5.9) == 0
    assert recorder.write(image, now=10.0 + interval * 6) == 1
    assert len(writer.frames) == 4
    assert recorder.fps == 25
    recorder.stop()


def test_timelapse_defaults_to_60_frames_per_minute(writer, tmp_path):
    recorder = VideoRecorder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.start(tmp_path / "lapse.mp4", image.shape, "timelapse", now=0)
    for index in range(1500):
        recorder.write(image, now=index / 25)
    assert recorder.frames_written == 60
    recorder.stop()


def test_rotation_keeps_output_size_and_image_aspect_ratio(writer, tmp_path):
    recorder = VideoRecorder()
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    recorder.start(tmp_path / "video.mp4", image.shape, "video", now=0)
    recorder.write(image, now=0)
    rotated = np.full((64, 48, 3), 200, dtype=np.uint8)
    recorder.write(rotated, now=0.04)
    result = writer.frames[-1]
    assert result.shape == image.shape
    assert np.all(result[:, :14] == 0)
    assert np.all(result[:, 14:50] == 200)
    assert np.all(result[:, 50:] == 0)
    recorder.stop()


def test_encoder_open_failure_releases_writer_and_keeps_capture_idle(writer, tmp_path):
    writer.isOpened.return_value = False
    recorder = VideoRecorder()
    with pytest.raises(OSError, match="Could not open MP4"):
        recorder.start(tmp_path / "video.mp4", (48, 64, 3), "video")
    assert not recorder.is_recording
    assert recorder.mode is None
    writer.release.assert_called_once()


def test_extension_handling_does_not_silently_replace_other_filename(writer, tmp_path):
    recorder = VideoRecorder()
    path = tmp_path / "video.mp4"
    path.write_bytes(b"existing video")
    with pytest.raises(FileExistsError):
        recorder.start(path.with_suffix(""), (48, 64, 3), "video")
    assert path.read_bytes() == b"existing video"
    with pytest.raises(ValueError, match=".mp4"):
        recorder.start(tmp_path / "video.png", (48, 64, 3), "video")
    recorder.start(tmp_path / "new video", (48, 64, 3), "video")
    assert recorder.path == tmp_path / "new video.mp4"
    recorder.stop()


@pytest.mark.parametrize("frames_per_minute", [0, -1, 1501, float("nan"), float("inf")])
def test_invalid_timelapse_rate_is_rejected(writer, tmp_path, frames_per_minute):
    recorder = VideoRecorder()
    with pytest.raises(ValueError, match="frames/minute"):
        recorder.start(
            tmp_path / "video.mp4", (48, 64, 3), "timelapse", frames_per_minute=frames_per_minute
        )
    assert not recorder.is_recording


@pytest.mark.parametrize("mode", ["video", "timelapse"])
def test_real_mp4_can_be_decoded_after_finalizing(tmp_path, mode):
    path = tmp_path / f"{mode}.mp4"
    recorder = VideoRecorder()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.start(path, image.shape, mode, now=0)
    for index in range(4):
        image[:] = (index * 50, 100, 200)
        recorder.write(image, now=index * (1.0 if mode == "timelapse" else 0.04))
    recorder.stop()
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == 25
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
        assert len(frames) == 4
        assert all(frame.shape == image.shape for frame in frames)
        assert np.allclose(frames[-1][24, 32], image[24, 32], atol=8)
    finally:
        capture.release()
