from dataclasses import dataclass, replace, field
import numpy as np
@dataclass
class PipelineConfig:
    n: int = 3000
    d: int = 500
    p: float = 0.1
    k: int = None
    Tr_target: float = None
    seed: int = 42
    params: dict=field(default_factory=dict)
BASE = PipelineConfig()
#params=dict(amps=(1/(3*(BASE.d)),1/(3*(BASE.d)),1/(3*(BASE.d))),freqs=(10, BASE.d**2/(3*(BASE.d))+10, 2*BASE.d**2/(3*(BASE.d))+10))
CURVE_OVERRIDES = {
    "sinusoid":    dict(k=6, params=dict(amps=(1,0.7,0.5), freqs=(2,5,10)), Tr_target=0.2),
    "trefoil":  dict(k=3, params=dict(scale=1), Tr_target=0.165),
    "segments": dict(k=2, params=dict(scale=1), Tr_target=0.05),
    "circle":   dict(k=2, params=dict(scale=1), Tr_target=0.15),
    "swiss":    dict(k=3, params=dict( scale=0.6, twists=3, height=0.4), Tr_target=0.17),
    "sphere":   dict(k=3, Tr_target=0.17),
    "spiral": dict(k=2, Tr_target=0.15, params=dict( scale=0.6, twists=3, height=0.4)),
    'nested_circles': dict(k=2, Tr_target=0.15, params=dict(radii=(0.1, 0.2, 0.3))),
    "trefoil2":  dict(k=3, params=dict(major_r=2.0, minor_r=1.0)),

}

def get_config(curve_name: str) -> PipelineConfig:
    return replace(BASE, **CURVE_OVERRIDES.get(curve_name, {}))