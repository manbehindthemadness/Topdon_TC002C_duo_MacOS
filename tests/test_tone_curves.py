import pytest

from topdon_duo.tone_curves import composite_curve


@pytest.mark.parametrize('gamma,boost,contrast,preset,expected', [
    (25, False, 50, 'balanced', [0, 18, 40, 96, 168, 210, 255]),
    (75, False, 50, 'balanced', [0, 106, 133, 177, 217, 237, 255]),
    (50, True, 50, 'balanced', [0, 27, 57, 128, 198, 228, 255]),
    (50, False, 75, 'balanced', [0, 20, 48, 128, 208, 235, 255]),
    (50, False, 50, 'shadow', [0, 18, 40, 96, 168, 210, 255]),
])
def test_native_curve_samples(gamma, boost, contrast, preset, expected):
    curve = composite_curve(gamma, boost, contrast, preset)
    assert [curve[i] for i in (0, 32, 64, 128, 192, 224, 255)] == expected
    assert len(curve) == 256 and all(0 <= v <= 255 for v in curve)
    assert curve == sorted(curve)


def test_neutral_preserves_identity():
    assert composite_curve(50, False, 50, 'balanced') == list(range(256))
