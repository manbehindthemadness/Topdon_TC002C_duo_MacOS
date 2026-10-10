"""
Preset baseline checks distinguish wire modes from gain-dependent processing banks.
"""

from unittest.mock import Mock, call

import pytest
from test_hardware_controls import controls as controls_fixture

from topdon_duo.camera import CameraError
from topdon_duo.hardware_controls import HardwareControls
from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.pipeline_hardware import PipelineHardware

controls = controls_fixture


def preset_hardware(request: pytest.FixtureRequest, mode: int = 1) -> HardwareControls:
    """
    Prepare a saved SDK mode with the independently tested ISP baseline supplied.
    """
    hardware: HardwareControls = request.getfixturevalue("controls")
    original = bytearray(hardware.original[2, 5])
    original[23] = mode
    hardware.original[2, 5] = bytes(original)
    hardware._fixed_range_baseline = Mock(return_value=(1000, 2800))
    return hardware


@pytest.mark.parametrize("initial_bank", [1, 4])
def test_preset_only_pipeline_applies_and_expansion_does_not_retry(
    request: pytest.FixtureRequest, initial_bank: int,
) -> None:
    """
    Apply Shadow with no other hardware nodes, then expand its UI without USB writes.
    """
    hardware = preset_hardware(request)
    bank = initial_bank

    def processing_command(mode: int | None = None) -> int:
        """
        Independently simulate the known wire-mode mapping for the current gain.
        """
        nonlocal bank
        if mode is not None:
            bank = {1: 1, 2: 2, 0: 3}[mode] + (3 if initial_bank == 4 else 0)
            return 1
        return bank

    command = Mock(side_effect=processing_command)
    hardware._processing_command = command
    document = default_pipeline()
    preset = node("hardware", "preset", value="shadow")
    document["hardware"] = [preset]
    document["software"] = [node("software", "source"), node("software", "output")]
    manager = PipelineHardware(hardware)
    manager.apply(document)
    assert hardware.processing_preset == "shadow"
    assert bank == (5 if initial_bank == 4 else 2)
    count = command.call_count
    preset["expanded"] = not preset["expanded"]
    manager.apply(document)
    assert command.call_count == count
    hardware.restore_processing_preset()
    assert bank == initial_bank


@pytest.mark.parametrize("bank", [1, 4])
@pytest.mark.parametrize("preset", ["shadow", "soft"])
def test_balanced_gain_banks_allow_first_preset_selection(
    request: pytest.FixtureRequest, bank: int, preset: str,
) -> None:
    """
    An otherwise empty pipeline can enable a preset from either Balanced gain bank.
    """
    hardware = preset_hardware(request)
    hardware._processing_command = Mock(return_value=bank)
    apply = Mock()
    hardware._apply_processing_preset = apply
    hardware.set_processing_preset(preset)
    apply.assert_called_once_with(preset)
    assert hardware.processing_preset == preset
    assert hardware._processing_preset_owned
    hardware.restore_processing_preset()
    assert apply.call_args_list == [call(preset), call("balanced")]
    assert not hardware._processing_preset_owned


@pytest.mark.parametrize("mode,bank", [(0, 1), (2, 4), (1, 0), (1, 7), (1, 8)])
def test_unsupported_baselines_report_actual_state_without_writes(
    request: pytest.FixtureRequest, mode: int, bank: int,
) -> None:
    """
    Retain the factory baseline check and expose the precise rejected mode/bank.
    """
    hardware = preset_hardware(request, mode)
    hardware._processing_command = Mock(return_value=bank)
    apply = Mock()
    hardware._apply_processing_preset = apply
    with pytest.raises(CameraError, match=f"startup AGC mode={mode}, current bank={bank}"):
        hardware.set_processing_preset("shadow")
    apply.assert_not_called()
    assert not hardware._processing_preset_owned


@pytest.mark.parametrize("preset,mode,bank", [("balanced", 1, 4), ("shadow", 2, 5),
                                             ("soft", 0, 6)])
def test_preset_verification_uses_wire_mode_and_gain_paired_bank(
    request: pytest.FixtureRequest, preset: str, mode: int, bank: int,
) -> None:
    """
    Setter uses legacy mode values; getter banks are only used for verification.
    """
    hardware: HardwareControls = request.getfixturevalue("controls")
    command = Mock(side_effect=[1, bank])
    hardware._processing_command = command
    hardware._apply_processing_preset(preset)
    assert command.call_args_list == [call(mode), call()]


@pytest.mark.parametrize("initial_bank,original", [(1, "balanced"), (2, "shadow"),
                                                  (3, "soft"), (4, "balanced"),
                                                  (5, "shadow"), (6, "soft")])
@pytest.mark.parametrize("requested", ["balanced", "shadow", "soft"])
def test_presets_switch_from_known_live_banks_and_restore_observed_preset(
    request: pytest.FixtureRequest, initial_bank: int, original: str, requested: str,
) -> None:
    """
    Handle the reported SDK mode1/live Soft bank3 mismatch, including selecting Balanced.
    """
    hardware = preset_hardware(request)
    hardware._processing_command = Mock(return_value=initial_bank)
    apply = Mock()
    hardware._apply_processing_preset = apply
    hardware.set_processing_preset(requested)
    apply.assert_called_once_with(requested)
    assert hardware.processing_preset == requested
    assert hardware._original_processing_preset == original
    hardware.restore_processing_preset()
    assert apply.call_args_list == [call(requested), call(original)]
    assert hardware.processing_preset == original
    assert not hardware._processing_preset_owned


def test_failed_selection_restores_soft_and_failed_cleanup_keeps_ownership(
    request: pytest.FixtureRequest,
) -> None:
    """
    Roll back a failed Shadow write to the observed Soft baseline, never a guessed bank.
    """
    hardware = preset_hardware(request)
    hardware._processing_command = Mock(return_value=3)
    apply = Mock(side_effect=[CameraError("write failed"), None])
    hardware._apply_processing_preset = apply
    with pytest.raises(CameraError, match="write failed"):
        hardware.set_processing_preset("shadow")
    assert apply.call_args_list == [call("shadow"), call("soft")]
    assert hardware.processing_preset == "soft" and not hardware._processing_preset_owned
    hardware._processing_preset_owned = True
    apply.side_effect = CameraError("restore failed")
    with pytest.raises(CameraError, match="restore failed"):
        hardware.restore_processing_preset()
    assert hardware._processing_preset_owned


def test_restoration_places_original_live_preset_after_sdk_refreshes(
    request: pytest.FixtureRequest,
) -> None:
    """
    Prevent SDK control restoration from overwriting the separately saved live preset.
    """
    hardware = preset_hardware(request)
    hardware._processing_command = Mock(return_value=3)
    operations: list[str] = []
    hardware._apply_processing_preset = Mock(side_effect=operations.append)
    hardware.set_processing_preset("shadow")
    hardware.set("contrast", 70, True)
    operations.clear()
    hardware.restore()
    assert operations == ["shadow", "soft"]
    assert hardware.processing_preset == "soft"


@pytest.mark.parametrize("start_empty", [False, True])
def test_first_pipeline_verifies_balanced_instead_of_trusting_default_label(
    request: pytest.FixtureRequest, start_empty: bool,
) -> None:
    """
    A Balanced-only pipeline must correct live Soft bank3 and later restore Soft.
    """
    hardware = preset_hardware(request)
    command = Mock(side_effect=[3, 1, 1, 1, 3])
    hardware._processing_command = command
    document = default_pipeline()
    document["software"] = [node("software", "source"), node("software", "output")]
    manager = PipelineHardware(hardware)
    if start_empty:
        manager.apply(document)
        command.assert_not_called()
    document["hardware"] = [node("hardware", "preset", value="balanced")]
    manager.apply(document)
    assert command.call_args_list == [call(), call(1), call()]
    assert hardware.processing_preset == "balanced"
    assert hardware._original_processing_preset == "soft"
    hardware.restore_processing_preset()
    assert command.call_args_list[-2:] == [call(0), call()]
    assert hardware.processing_preset == "soft"


@pytest.mark.parametrize("initial_bank,original", [(2, "shadow"), (3, "soft")])
def test_first_pipeline_saves_live_preset_before_sdk_writes(
    request: pytest.FixtureRequest, initial_bank: int, original: str,
) -> None:
    """
    Preserve the startup bank when a combined preset/control edit reloads the SDK bank.
    """
    hardware = preset_hardware(request)
    bank = initial_bank
    write = hardware.write

    def sdk_write(key: tuple[int, int], payload: bytes) -> None:
        """
        Model the SDK refresh independently of the preset setter.
        """
        nonlocal bank
        write(key, payload)
        if key[0] == 2:
            bank = 1

    def processing_command(mode: int | None = None) -> int:
        """
        Apply the audited wire-mode mapping to the simulated live bank.
        """
        nonlocal bank
        if mode is not None:
            bank = {1: 1, 2: 2, 0: 3}[mode]
            return 1
        return bank

    hardware.write = Mock(side_effect=sdk_write)
    hardware._processing_command = Mock(side_effect=processing_command)
    document = default_pipeline()
    document["hardware"] = [
        node("hardware", "contrast", value=70),
        node("hardware", "preset", value="balanced"),
    ]
    PipelineHardware(hardware).apply(document)
    assert hardware._original_processing_preset == original
    assert bank == 1
    hardware.restore()
    assert bank == initial_bank
    assert hardware.processing_preset == original


def test_add_balanced_preset_while_fixed_detail_is_active(
    request: pytest.FixtureRequest,
) -> None:
    """
    Release Fixed Detail around explicit preset verification and re-enable it afterward.
    """
    hardware = preset_hardware(request)
    hardware._processing_command = Mock(return_value=1)
    hardware._fixed_range_bounds = (1000, 2800)
    hardware._fixed_range_command = Mock()
    document = default_pipeline()
    document["hardware"] = [node("hardware", "detail", enabled=True, amount=60, fixed=True)]
    manager = PipelineHardware(hardware)
    manager.apply(document)
    assert hardware.fixed_range
    hardware._fixed_range_command.reset_mock()
    document["hardware"].insert(0, node("hardware", "preset", value="balanced"))
    manager.apply(document)
    assert hardware.fixed_range and hardware._fixed_range_owned
    assert hardware.processing_preset == "balanced"
    assert manager.document == document
    assert hardware._fixed_range_command.call_args_list == [
        call(1000, 2800), call(0, 0), call(0, 16383), call(0, 16383),
    ]
    hardware._processing_command.reset_mock()
    document["hardware"][0]["expanded"] = False
    manager.apply(document)
    hardware._processing_command.assert_not_called()
