"""
Portable custom packages, validated configuration and worker execution boundaries.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline

from topdon_duo.custom_nodes.bundle import load_folder, package_files, read_json_object
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.processing.branch import BranchProcessor


@pytest.mark.parametrize(
    "files",
    [
        {"helper.py": ""},
        {"__init__.py": None},
        {"__init__.py": "def broken(:"},
        {"__init__.py": "", "config.json": "[] trailing"},
        {"__init__.py": "", "config.json": '{"x": 1e999}'},
        {"__init__.py": "", "helper.py": "", "helper.py/data.json": "{}"},
    ],
)
def test_package_document_validation_rejects_malformed_files(files: dict[str, Any]) -> None:
    """
    Reject invalid custom package data before it reaches a worker.
    """
    item = node("software", "custom", name="module", package=json.dumps(files))
    with pytest.raises(ValueError):
        validate_pipeline(raw_pipeline(item))


def test_folder_refuses_symlinks_and_missing_entrypoint(tmp_path: Path) -> None:
    """
    Reject incomplete packages and links to files outside their folder.
    """
    (tmp_path / "helper.py").write_text("")
    with pytest.raises(ValueError, match="__init__"):
        load_folder(tmp_path)
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "link.py").symlink_to(tmp_path / "helper.py")
    with pytest.raises(ValueError, match="symbolic links"):
        load_folder(tmp_path)


@pytest.mark.parametrize("large_resource", [False, True])
def test_large_packages_survive_folder_loading_and_document_round_trip(
    tmp_path: Path, large_resource: bool,
) -> None:
    """
    Execute portable packages exceeding the former byte or file-count limits.
    """
    (tmp_path / "__init__.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def process(image, config):\n"
        " data = json.loads(Path(__file__).with_name('data.json').read_text())\n"
        " return image + data['amount']\n"
    )
    resource = {"amount": 12, "padding": "x" * (2 * 1024 * 1024) if large_resource else ""}
    (tmp_path / "data.json").write_text(json.dumps(resource))
    if not large_resource:
        for index in range(128):
            (tmp_path / f"helper_{index}.py").write_text("")
    item = node("software", "custom", **load_folder(tmp_path))
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    processor = CustomProcessor()
    try:
        result = processor.apply(np.zeros((3, 4, 3)), document["software"][2])
        assert np.all(result == 12)
    finally:
        processor.close()


def custom_item(source: str, config: str = "{}") -> dict[str, Any]:
    """
    Construct one portable custom node with a small in-memory package.
    """
    item = node(
        "software",
        "custom",
        name="example",
        package=json.dumps({"__init__.py": source}),
        config=config,
    )
    return item


def test_folder_snapshot_relative_import_config_and_portable_round_trip(tmp_path: Path) -> None:
    """
    Retain helpers and JSON data even when the original selected folder is removed.
    """
    folder = tmp_path / "custom"
    folder.mkdir()
    (folder / "__init__.py").write_text(
        "from .helper import apply\ndef process(image, config): return apply(image, config)\n"
    )
    (folder / "helper.py").write_text("def apply(image, config): return image + config['amount']\n")
    (folder / "config.json").write_text('{"amount": 12}')
    params = load_folder(folder)
    item = node("software", "custom", **params)
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    processor = CustomProcessor()
    try:
        result = processor.apply(np.zeros((3, 4, 3), dtype=np.float32), document["software"][2])
        assert np.all(result == 12)
        assert result.dtype == np.float32
        package = processor.packages[item["id"]]
        module_name, temporary_folder = package.name, package.directory.name
        assert module_name + ".helper" in sys.modules
    finally:
        processor.close()
    assert module_name not in sys.modules
    assert module_name + ".helper" not in sys.modules
    assert not Path(temporary_folder).exists()


@pytest.mark.parametrize("text", ["[]", "null", "{", '{"x": NaN}', '{"x": 1e999}', "{}" * 40000])
def test_configuration_rejects_invalid_nonfinite_or_oversized_json(text: str) -> None:
    """
    Only bounded JSON objects with finite values can become configuration.
    """
    with pytest.raises(ValueError):
        read_json_object(text)


@pytest.mark.parametrize(
    "name", ["../x.py", "/x.py", "a/../../x.py", "a\\x.py", "a//x.py", "x.txt"]
)
def test_package_rejects_unsafe_paths(name: str) -> None:
    """
    Imported package paths stay within their own temporary directory.
    """
    with pytest.raises(ValueError):
        package_files(json.dumps({"__init__.py": "", name: ""}))


def test_folder_validation_does_not_execute_code(tmp_path: Path) -> None:
    """
    Browsing and importing pipeline documents never execute module-level code.
    """
    marker = tmp_path / "marker"
    (tmp_path / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "def process(image, config): return image\n"
    )
    validate_pipeline(raw_pipeline(node("software", "custom", **load_folder(tmp_path))))
    assert not marker.exists()


@pytest.mark.parametrize(
    "source", ["raise RuntimeError('bad import')", "raise SystemExit(2)", "process = 3"]
)
def test_import_errors_are_reported_and_modules_released(source: str) -> None:
    """
    User imports cannot leak temporary namespaces or terminate the viewer through SystemExit.
    """
    before = {name for name in sys.modules if name.startswith("_topdon_custom_")}
    processor = CustomProcessor()
    with pytest.raises(ValueError, match="Custom example"):
        processor.apply(np.zeros((2, 2, 3)), custom_item(source))
    assert not processor.packages
    assert {name for name in sys.modules if name.startswith("_topdon_custom_")} == before


@pytest.mark.parametrize(
    "expression",
    [
        "None",
        "np.zeros((2, 2))",
        "np.full((2, 2, 3), np.nan)",
        "np.zeros((0, 2, 3))",
        "np.zeros((2001, 2000, 3), dtype=np.uint8)",
        "np.zeros((2, 2, 3), dtype=complex)",
    ],
)
def test_invalid_custom_outputs_become_pipeline_errors(expression: str) -> None:
    """
    Reject wrong shapes, nonnumeric/nonfinite values and oversized display frames.
    """
    processor = CustomProcessor()
    try:
        item = custom_item(f"import numpy as np\ndef process(image, config): return {expression}\n")
        with pytest.raises(ValueError, match="4 megapixel"):
            processor.apply(np.zeros((2, 2, 3)), item)
    finally:
        processor.close()


def test_package_cache_config_updates_reload_and_input_ownership() -> None:
    """
    Cache imports, accept new config each call, reload code and isolate input mutations.
    """
    source = (
        "count = 0\ndef process(image, config):\n"
        " global count\n count += 1\n image[:] = count + config['value']\n return image\n"
    )
    processor = CustomProcessor()
    item = custom_item(source, '{"value": 5}')
    image = np.zeros((2, 2, 3), dtype=np.float32)
    try:
        assert np.all(processor.apply(image, item) == 6)
        item["params"]["config"] = '{"value": 10}'
        assert np.all(processor.apply(image, item) == 12)
        assert not image.any()
        old_name = processor.packages[item["id"]].name
        item["params"]["package"] = json.dumps(
            {"__init__.py": "def process(image, config): return image + 999"}
        )
        assert np.all(processor.apply(image, item) == 255)
        assert old_name not in sys.modules
        processor.retain(set())
        assert not processor.packages
    finally:
        processor.close()


def test_callback_errors_and_cleared_package_cleanup() -> None:
    """
    Report user callback failures and unload a cleared custom package.
    """
    processor = CustomProcessor()
    item = custom_item("def process(image, config): raise RuntimeError('callback failed')")
    try:
        with pytest.raises(ValueError, match="callback failed"):
            processor.apply(np.zeros((2, 2, 3)), item)
        name = processor.packages[item["id"]].name
        item["params"]["package"] = "{}"
        image = np.zeros((2, 2, 3))
        assert processor.apply(image, item) is image
        assert name not in sys.modules
    finally:
        processor.close()


def test_each_node_owns_its_package_globals() -> None:
    """
    Identical packages in different nodes have independent Python state.
    """
    source = (
        "count = 0\ndef process(image, config):\n global count\n count += 1\n return image + count"
    )
    processor = CustomProcessor()
    first, second = custom_item(source), custom_item(source)
    try:
        image = np.zeros((2, 2, 3))
        assert np.all(processor.apply(image, first) == 1)
        assert np.all(processor.apply(image, first) == 2)
        assert np.all(processor.apply(image, second) == 1)
    finally:
        processor.close()


def test_branch_custom_order_bypass_and_raw_measurement_ownership() -> None:
    """
    Run custom nodes in order and leave captured radiometry untouched.
    """
    processor = BranchProcessor(apple_available=False, nvidia_available=False)
    item = custom_item("def process(image, config):\n image[:] = 100\n return image\n")
    frame = make_frame()
    document = raw_pipeline(item, node("software", "brightness", amount=10))
    try:
        output, source = processor.process(frame, None, document)
        assert source == "raw"
        assert np.allclose(output, 125.5)
        assert frame == make_frame()
        item["bypass"] = True
        processor.process(frame, None, document)
        assert not processor.custom.packages
    finally:
        processor.close()
