import hashlib
import io
import ssl
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from topdon_duo import onnx_models
from topdon_duo.enhancement_limits import enhancement_pass_limits
from topdon_duo.onnx_upsampling import ONNXRuntime, ONNXUpsampler
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_titles import node_title


def fake_runtime(monkeypatch, bad_output=False, fixed_shape=False):
    calls = []
    monkeypatch.setattr("topdon_duo.onnx_upsampling.verified_model", lambda model: model.encode())
    monkeypatch.setattr("topdon_duo.onnx_upsampling.explicit_coreml_padding", lambda data: data)
    monkeypatch.setattr(
        "topdon_duo.onnx_upsampling.coreml_input_shape_overrides",
        lambda data, shape, **kwargs: {
            "batch": shape[0],
            "height": shape[2],
            "width": shape[3],
        },
    )

    class Options:
        def __init__(self):
            self.overrides = {}

        def add_free_dimension_override_by_name(self, name, value):
            self.overrides[name] = value

    class Session:
        def __init__(self, data, **kwargs):
            self.model = data.decode()
            self.providers = kwargs["providers"]
            calls.append(kwargs)

        def disable_fallback(self):
            pass

        def get_providers(self):
            return [p[0] if isinstance(p, tuple) else p for p in self.providers]

        def get_inputs(self):
            inputs = [SimpleNamespace(name="x", shape=(1, 1, 224, 224) if fixed_shape else ())]
            if self.model == "ffdnet-gray":
                inputs.append(SimpleNamespace(name="sigma", shape=(1, 1, 1, 1)))
            return inputs

        def run(self, _outputs, feed):
            blob = feed["x"]
            calls.append(blob.copy())
            if self.model == "ffdnet-gray":
                calls[-1] = (blob.copy(), feed["sigma"].copy())
            factor = onnx_models.MODELS[self.model]["factor"]
            output = np.repeat(np.repeat(blob, factor, axis=2), factor, axis=3)
            if bad_output:
                output[:] = np.nan
            return [output]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=Options,
            InferenceSession=Session,
            get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    return calls


@pytest.mark.parametrize(
    "model,factor", [(name, spec["factor"]) for name, spec in onnx_models.MODELS.items()]
)
@pytest.mark.parametrize("color", [False, True])
def test_inference_shapes_color_order_session_reuse_and_input_ownership(
    monkeypatch, model, factor, color
):
    calls = fake_runtime(monkeypatch)
    image = np.full((4, 6, 3) if color else (4, 6), 100, dtype=np.uint8)
    if color:
        image[:] = [10, 70, 200]
    original = image.copy()
    runtime = ONNXRuntime()
    result = runtime.apply(image, model, amount=0.5)
    assert result.shape == ((4 * factor, 6 * factor, 3) if color else (4 * factor, 6 * factor))
    assert result.dtype == np.uint8
    assert np.array_equal(image, original)
    assert (
        np.max(np.abs(result.astype(int) - cv2.resize(image, (6 * factor, 4 * factor)).astype(int)))
        <= 2
    )
    if onnx_models.MODELS[model]["rgb"] and color:
        assert np.allclose(calls[1][0, :, 0, 0], [200, 70, 10] / np.array(255.0))
    runtime.apply(image, model)
    assert len(calls) == 3  # One session, two inferences.


def test_fixed_espcn_tiles_preserve_rectangular_geometry_and_cover_seams(monkeypatch):
    calls = fake_runtime(monkeypatch, fixed_shape=True)
    image = np.random.default_rng(10).integers(0, 256, (192, 256), dtype=np.uint8)
    output = ONNXRuntime().apply(image, "espcn")
    assert np.array_equal(output, np.repeat(np.repeat(image, 3, axis=0), 3, axis=1))
    assert len(calls) == 3  # Session plus two overlapping 224x224 tiles.
    assert all(blob.shape == (1, 1, 224, 224) for blob in calls[1:])


def test_ffdnet_odd_geometry_noise_normalization_and_session_reuse(monkeypatch):
    calls = fake_runtime(monkeypatch)
    image = np.random.default_rng(8).integers(0, 256, (5, 7), dtype=np.uint8)
    runtime = ONNXRuntime()
    assert np.array_equal(runtime.apply(image, "ffdnet-gray", noise=15), image)
    assert calls[1][0].shape == (1, 1, 6, 8)
    assert np.allclose(calls[1][1], 15 / 255)
    assert np.allclose(calls[1][0][0, 0, -1], calls[1][0][0, 0, -2])
    runtime.apply(image, "ffdnet-gray", noise=75)
    assert len(calls) == 3
    assert np.allclose(calls[2][1], 75 / 255)
    for invalid in (-1, 76, np.nan):
        with pytest.raises(ValueError, match="sigma"):
            runtime.apply(image, "ffdnet-gray", noise=invalid)


@pytest.mark.parametrize("model", ["dncnn-25", "ffdnet-gray", "realesr-general-x4v3"])
def test_new_models_apple_shape_specialization(monkeypatch, model):
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    ONNXRuntime().apply(np.zeros((5, 7, 3), np.uint8), model, "coreml")
    shape = (6, 8) if model == "ffdnet-gray" else (5, 7)
    assert calls[0]["sess_options"].overrides == {"batch": 1, "height": shape[0], "width": shape[1]}
    assert calls[0]["providers"][0][1]["RequireStaticInputShapes"] == "1"


def test_denoiser_download_needs_export_and_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    opener = Mock(side_effect=AssertionError("no network"))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError, match="export_visual_denoisers.py dncnn-25"):
        onnx_models.download_model("dncnn-25")
    with pytest.raises(ValueError, match="export_visual_denoisers.py ffdnet-gray"):
        onnx_models.verified_model("ffdnet-gray")
    opener.assert_not_called()


@pytest.mark.parametrize("apple_available,expected", [(False, "cpu"), (True, "coreml")])
def test_denoiser_pipeline_retains_size_preferences_measurements_and_closes(
    monkeypatch, apple_available, expected
):
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls, engines = [], []

    class Engine:
        def __init__(self):
            self.closed = False
            engines.append(self)

        def apply(self, image, model, backend, compute, amount, noise=15):
            calls.append((image.shape, model, backend, compute, amount, noise))
            return image.copy()

        def close(self):
            self.closed = True

    monkeypatch.setattr("topdon_duo.pipeline_processing.ONNXUpsampler", Engine)
    denoiser = node("software", "onnx_denoise", backend="coreml", noise=32)
    document = raw_pipeline(denoiser)
    original = deepcopy(document)
    assert validate_pipeline(document) == document
    assert "sigma 32" in node_title("software", denoiser)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=4)
        assert output.shape == (768, 1024, 3)
        assert calls[0][0][:2] == (192, 256)
        assert calls[0][2] == expected
        assert calls[0][-1] == 32
        assert document == original
        assert np.array_equal(measurements, before)
        denoiser["bypass"] = True
        processor.process(frame, measurements, document)
        assert engines[0].closed
        assert len(calls) == 1
    finally:
        processor.close()


def test_apple_provider_configuration_and_invalid_output(monkeypatch):
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    ONNXRuntime().apply(np.zeros((4, 6), np.uint8), "espcn", "coreml", "ALL")
    assert calls[0]["providers"][0][1]["MLComputeUnits"] == "ALL"
    fake_runtime(monkeypatch, bad_output=True)
    with pytest.raises(ValueError, match="non-finite"):
        ONNXRuntime().apply(np.zeros((4, 6), np.uint8), "mewzoom")


@pytest.mark.parametrize("model", ["mewzoom-v1-2x", "mewzoom-v1-4x"])
def test_v1_apple_shape_specialization_and_size_change_reload(monkeypatch, model):
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    runtime = ONNXRuntime()
    runtime.apply(np.zeros((4, 6, 3), np.uint8), model, "coreml")
    assert calls[0]["providers"][0][1]["RequireStaticInputShapes"] == "1"
    assert calls[0]["sess_options"].overrides == {"batch": 1, "height": 4, "width": 6}
    runtime.apply(np.zeros((4, 6, 3), np.uint8), model, "coreml")
    assert len(calls) == 3  # Reuse one session for matching dimensions.
    runtime.apply(np.zeros((8, 10, 3), np.uint8), model, "coreml")
    assert calls[3]["sess_options"].overrides == {"batch": 1, "height": 8, "width": 10}
    assert len(calls) == 5


def test_v1_cpu_keeps_dynamic_shapes_and_no_apple_overrides(monkeypatch):
    calls = fake_runtime(monkeypatch)
    runtime = ONNXRuntime()
    runtime.apply(np.zeros((4, 6), np.uint8), "mewzoom-v1-2x")
    runtime.apply(np.zeros((8, 10), np.uint8), "mewzoom-v1-2x")
    assert len(calls) == 3
    assert calls[0]["sess_options"].overrides == {}
    assert calls[0]["providers"] == ["CPUExecutionProvider"]


def test_zero_amount_and_pixel_budget_do_not_load_models(monkeypatch):
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("topdon_duo.onnx_upsampling.verified_model", load)
    image = np.zeros((768, 1024), np.uint8)
    assert ONNXRuntime().apply(image, "mewzoom", amount=0) is image
    assert ONNXUpsampler().apply(image, "mewzoom", amount=0) is image
    with pytest.raises(ValueError, match="4 megapixels"):
        ONNXRuntime().apply(image, "mewzoom")
    load.assert_not_called()


def test_verified_download_is_atomic_reused_and_rejects_wrong_content(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    data = b"test model fixture"
    spec = dict(onnx_models.MODELS["espcn"], sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(onnx_models.MODELS, "espcn", spec)
    opener = Mock(side_effect=lambda *_args, **_kwargs: io.BytesIO(data))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    path = onnx_models.download_model("espcn")
    assert onnx_models.verified_model("espcn") == data
    assert onnx_models.download_model("espcn") == path
    assert opener.call_count == 1
    context = opener.call_args.kwargs["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    monkeypatch.setitem(onnx_models.MODELS, "mewzoom", dict(spec, sha256="0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        onnx_models.download_model("mewzoom")
    assert not onnx_models.model_path("mewzoom").exists()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "platform,custom_ca", [("darwin", False), ("darwin", True), ("linux", False)]
)
def test_download_trust_bundle_is_mac_only_and_respects_custom_ca(monkeypatch, platform, custom_ca):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    if custom_ca:
        monkeypatch.setenv("SSL_CERT_FILE", "/custom/trusted-ca.pem")
    context = Mock()
    monkeypatch.setattr(onnx_models.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(onnx_models.Path, "is_file", lambda _path: True)
    assert onnx_models.download_ssl_context() is context
    if platform == "darwin" and not custom_ca:
        context.load_verify_locations.assert_called_once_with(cafile="/etc/ssl/cert.pem")
    else:
        context.load_verify_locations.assert_not_called()


def test_v1_installer_verifies_publisher_pointer_and_retains_checksum_offline(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    data = b"v1 test model"
    checksum = hashlib.sha256(data).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{checksum}\nsize {len(data)}\n".encode()
    opener = Mock(side_effect=[io.BytesIO(pointer), io.BytesIO(data)])
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    path = onnx_models.download_model("mewzoom-v1-2x")
    assert "/raw/main/model.onnx" in opener.call_args_list[0].args[0]
    assert "/resolve/main/model.onnx" in opener.call_args_list[1].args[0]
    assert path.with_suffix(".sha256").read_text().strip() == checksum
    assert onnx_models.verified_model("mewzoom-v1-2x") == data
    assert onnx_models.download_model("mewzoom-v1-2x") == path
    assert opener.call_count == 2  # Cached weights need no remote metadata request.


@pytest.mark.parametrize(
    "pointer",
    [
        b"not an LFS pointer",
        b"version https://git-lfs.github.com/spec/v1\noid sha256:"
        + b"a" * 64
        + b"\nsize 999999999\n",
    ],
)
def test_v1_installer_rejects_invalid_or_oversized_publisher_metadata(
    monkeypatch, tmp_path, pointer
):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    opener = Mock(return_value=io.BytesIO(pointer))
    monkeypatch.setattr(onnx_models.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError):
        onnx_models.download_model("mewzoom-v1-4x")
    assert opener.call_count == 1
    assert list(tmp_path.iterdir()) == []


def test_missing_model_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="uv run topdon-duo-models mewzoom"):
        onnx_models.verified_model("mewzoom")


def test_node_roundtrip_title_and_downstream_acnet_size_limits():
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == 1
    sr["bypass"] = True
    assert enhancement_pass_limits(document)[ai["id"]] == 3


def test_v0_2x_node_is_portable_and_retains_existing_v0_4x_selection():
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model="mewzoom-v0-2x", backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom V0 2×" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == 2
    assert node("software", "onnx_superresolution", model="mewzoom")["params"]["model"] == "mewzoom"


@pytest.mark.parametrize("model,maximum", [("mewzoom-v1-2x", 2), ("mewzoom-v1-4x", 1)])
def test_v1_node_roundtrip_and_factor_dependent_limits(model, maximum):
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model=model, backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom V1" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == maximum


@pytest.mark.parametrize("apple_available,expected", [(False, "cpu"), (True, "coreml")])
def test_pipeline_cpu_fallback_retains_preferences_and_measurements(
    monkeypatch, apple_available, expected
):
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls = []
    engines = []

    class Engine:
        def __init__(self):
            self.closed = False
            engines.append(self)

        def apply(self, image, model, backend, compute, amount):
            calls.append((image.shape, model, backend, compute, amount))
            factor = onnx_models.MODELS[model]["factor"]
            return cv2.resize(image, (image.shape[1] * factor, image.shape[0] * factor))

        def close(self):
            self.closed = True

    monkeypatch.setattr("topdon_duo.pipeline_processing.ONNXUpsampler", Engine)
    sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
    document = raw_pipeline(sr)
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=4)
        assert output.shape == (768, 1024, 3)
        assert calls[0][0][:2] == (192, 256)
        assert calls[0][2] == expected
        assert document == original
        assert np.array_equal(measurements, before)
        sr["bypass"] = True
        processor.process(frame, measurements, document)
        assert len(calls) == 1
        assert engines[0].closed
    finally:
        processor.close()


def test_helper_crash_is_contained_and_configuration_change_clears_error(monkeypatch):
    process = Mock()
    process.is_alive.return_value = False
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.side_effect = EOFError
    engine = ONNXUpsampler()
    engine.key = "espcn", "cpu", "CPUAndGPU"
    engine.process, engine.connection = process, connection
    with pytest.raises(ValueError, match="helper exited"):
        engine.apply(np.zeros((4, 4), np.uint8), "espcn")
    assert engine.process is None
    process.close.assert_called_once()


@pytest.mark.parametrize("model", tuple(onnx_models.MODELS))
def test_real_models_if_installed(model):
    """Opt-in local smoke test; no downloads/network in the test suite."""
    if not onnx_models.model_path(model).exists():
        pytest.skip("Optional visual model weights not installed")
    spec = onnx_models.MODELS[model]
    engine = ONNXUpsampler()
    image = np.full((24, 32, 3), 100, np.uint8)
    try:
        output = engine.apply(image, model)
        assert output.shape == (24 * spec["factor"], 32 * spec["factor"], 3)
    finally:
        engine.close()


@pytest.mark.parametrize("model", ["mewzoom-v1-2x", "mewzoom-v1-4x"])
def test_real_v1_shape_overrides_preserve_cpu_outputs(model):
    from topdon_duo.coreml_model import coreml_input_shape_overrides

    if not onnx_models.model_path(model).exists():
        pytest.skip("V1 weights not installed")
    ort = pytest.importorskip("onnxruntime")
    data = onnx_models.verified_model(model)
    shape = (1, 3, 24, 32)
    overrides = coreml_input_shape_overrides(data, shape)
    assert set(overrides.values()) == {1, 24, 32}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    dynamic = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    for name, value in overrides.items():
        options.add_free_dimension_override_by_name(name, value)
    fixed = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    blob = np.random.default_rng(40).random(shape, dtype=np.float32)
    assert np.allclose(
        dynamic.run(None, {"x": blob})[0], fixed.run(None, {"x": blob})[0], atol=1e-6
    )


@pytest.mark.parametrize("model", ["mewzoom-v1-2x", "mewzoom-v1-4x"])
def test_real_v1_coreml_initialization_no_axis_error(model):
    if sys.platform != "darwin" or not onnx_models.model_path(model).exists():
        pytest.skip("Requires macOS and installed V1 weights")
    engine = ONNXUpsampler()
    try:
        output = engine.apply(np.zeros((24, 32, 3), np.uint8), model, "coreml")
        factor = onnx_models.MODELS[model]["factor"]
        assert output.shape == (24 * factor, 32 * factor, 3)
    except ValueError as exc:
        if "Failed to create a working directory appropriate for URL" in str(exc):
            pytest.skip("Core ML compilation temp directory is blocked by this sandbox")
        raise
    finally:
        engine.close()


def test_popup_apple_controls_and_cpu_preference_retention(tmp_path):
    import subprocess

    from test_capture_panel import popup_environment

    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
window = ViewWindow(lambda message: None)
editor = window.pipeline_editor
sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
editor.document["software"].insert(-1, sr)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[sr["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["backend"].isHidden()
assert sr["params"]["backend"] == "coreml"
assert sr["params"]["model"] == "mewzoom"
assert "passes" not in rows
denoise = node("software", "onnx_denoise", model="ffdnet-gray", backend="coreml", noise=42)
editor.document["software"].insert(-1, denoise)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[denoise["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
assert not rows["noise"].isHidden()
denoise["params"]["model"] = "dncnn-25"
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["noise"].isHidden()
assert rows["backend"].isHidden()
assert denoise["params"]["backend"] == "coreml"
assert denoise["params"]["noise"] == 42
window.close()
"""
    env = popup_environment()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
