"""Duo composite tone curves, matched to the native integer LUT builder.

Functional lookup data from the verified Duo2.09.031 curve routines; no firmware
image, register discovery or persistent device writes are used at runtime.
"""

GAMMA_LOW = bytes.fromhex(
    "000000000000000000000000010101010101010102020202020203030303040404040505050506060607070708080809"
    "09090a0a0b0b0b0c0c0d0d0e0e0f0f1010111112121313141415151617171818191a1a1b1c1c1d1e1e1f202021222323"
    "242526262728292a2a2b2c2d2e2f2f30313233343536373838393a3b3c3d3e3f4041424344454647494a4b4c4d4e4f50"
    "51525455565758595b5c5d5e5f61626364666768696b6c6d6f70717374757778797b7c7e7f808283858688898b8c8e8f"
    "9192949597989a9b9d9ea0a2a3a5a6a8aaabadafb0b2b4b5b7b9babcbec0c1c3c5c7c8cacccecfd1d3d5d7d9dadcdee0"
    "e2e4e6e8e9ebedeff1f3f5f7f9fbfdff"
)

GAMMA_HIGH = bytes.fromhex(
    "0065727a8084898c8f929597999b9d9fa1a2a4a5a7a8aaabacadaeafb0b1b2b3b4b5b6b7b8b9bababbbcbdbebebfc0c0"
    "c1c2c2c3c4c4c5c5c6c7c7c8c8c9c9cacbcbcccccdcdcececfcfcfd0d0d1d1d2d2d3d3d3d4d4d5d5d6d6d6d7d7d8d8d8"
    "d9d9d9dadadbdbdbdcdcdcdddddddedededfdfdfe0e0e0e1e1e1e2e2e2e2e3e3e3e4e4e4e4e5e5e5e6e6e6e6e7e7e7e8"
    "e8e8e8e9e9e9e9eaeaeaeaebebebebececececededededeeeeeeeeefefefefeff0f0f0f0f1f1f1f1f2f2f2f2f2f3f3f3"
    "f3f3f4f4f4f4f4f5f5f5f5f5f6f6f6f6f6f7f7f7f7f7f8f8f8f8f8f9f9f9f9f9fafafafafafafbfbfbfbfbfcfcfcfcfc"
    "fcfdfdfdfdfdfdfefefefefefeffffff"
)

CONTRAST_LOW = bytes.fromhex(
    "4a4b4c4d4e4f5051525354555657595a5b5b5c5d5e5f5f6061616263636465656666676768686969696a6a6b6b6c6c6c"
    "6d6d6d6e6e6f6f6f70707070717171727272737373737474747575757576767676777777777878787878797979797a7a"
    "7a7a7a7b7b7b7b7b7c7c7c7c7c7d7d7d7d7d7e7e7e7e7e7e7f7f7f7f7f7f808080808080808080808181818181818282"
    "82828283838383838484848484858585858586868686878787878788888888898989898a8a8a8a8b8b8b8c8c8c8c8d8d"
    "8d8e8e8e8f8f8f8f9090909191929292939393949495959696969797989899999a9a9b9c9c9d9e9e9fa0a0a1a2a3a4a4"
    "a5a6a8a9aaabadaeb0b1b2b3b4b5b6b7"
)

CONTRAST_HIGH = bytes.fromhex(
    "00000000000000000101010101010202020203030303040405050506060707080809090a0a0b0b0c0d0d0e0e0f101111"
    "121314141516171819191a1b1c1d1e1f2021222324252627292a2b2c2d2e30313233353637383a3b3d3e3f4142444547"
    "484a4b4d4e505153555658595b5d5f6062646667696b6d6f71727476787a7c7e80838587898b8d8e9092949698999b9d"
    "9fa0a2a4a6a7a9aaacaeafb1b2b4b5b7b8babbbdbec0c1c2c4c5c7c8c9cacccdcecfd1d2d3d4d5d6d8d9dadbdcdddedf"
    "e0e1e2e3e4e5e6e6e7e8e9eaebebecedeeeeeff0f1f1f2f2f3f4f4f5f5f6f6f7f7f8f8f9f9fafafafbfbfcfcfcfcfdfd"
    "fdfdfefefefefefeffffffffffffffff"
)

def blend(values, low, high, strength):
    if strength <= 50:
        return [(low[v] * (50 - strength) + v * strength) // 50 for v in values]
    return [(v * (100 - strength) + high[v] * (strength - 50)) // 50 for v in values]


def composite_curve(gamma: int, boost: bool, contrast: int, preset: str) -> list[int]:
    """50 is a neutral additional gamma adjustment; retain native preset gamma."""
    values = list(range(256))
    if boost:
        values = blend(values, CONTRAST_LOW, CONTRAST_HIGH, 60)
    values = blend(values, GAMMA_LOW, GAMMA_HIGH, 25 if preset == "shadow" else 50)
    values = blend(values, GAMMA_LOW, GAMMA_HIGH, gamma)
    return blend(values, CONTRAST_LOW, CONTRAST_HIGH, contrast)
