import numpy as np

class Curve:
    """curve class: stores F, dF, ambient dimension."""
    def __init__(self, F, dF, ambient_dim):
        self.F_func = F
        self.dF_func = dF
        self.ambient_dim = ambient_dim

    def F(self, t):
        return self.F_func(t)

    def dF(self, t):
        return self.dF_func(t)

    def speed(self, t):
        return np.linalg.norm(self.dF(t), axis=1)
    
def make_sinusoid(amps=(1.0, 0.7, 0.5), freqs=(1, 4, 12)) -> Curve:
    def sinusoid_F(t):
        t = np.asarray(t).flatten()
        out = np.zeros((len(t), 2 * len(amps)))
        for i, (a, w) in enumerate(zip(amps, freqs)):
            out[:, 2*i]   = a * np.cos(2 * np.pi * w * t)
            out[:, 2*i+1] = a * np.sin(2 * np.pi * w * t)
        return out

    def sinusoid_dF(t):
        t = np.asarray(t).flatten()
        out = np.zeros((len(t), 2 * len(amps)))
        for i, (a, w) in enumerate(zip(amps, freqs)):
            c = 2 * np.pi * w
            out[:, 2*i]   = -a * c * np.sin(c * t)
            out[:, 2*i+1] =  a * c * np.cos(c * t)
        return out

    return Curve(F=sinusoid_F, dF=sinusoid_dF, ambient_dim=2 * len(amps))

def make_interval(a, b) -> Curve:
    """Segment from point a to point b (each length-D arrays)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    def interval_F(t):
        t = np.asarray(t).flatten()
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a[None, :] + t[:, None] * (b - a)[None, :]
    
    def interval_dF(t):
        t = np.asarray(t).flatten()
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return np.tile((b - a)[None, :], (len(t), 1))
    
    return Curve(F=interval_F( a, b), dF=interval_dF( a, b), ambient_dim=1)

def make_trefoil(scale, scale2, scale3):
    def trefoil_F(t):
        t = np.asarray(t).flatten()
        u = 2 * np.pi * t
        out = np.zeros((len(t), 3))
        out[:, 0] = scale * (np.sin(u)   + 2 * np.sin(2 * u))
        out[:, 1] = scale2 * (np.cos(u)   - 2 * np.cos(2 * u))
        out[:, 2] = -scale3 * np.sin(3 * u)
        return out
    
    def trefoil_dF(t):
        t = np.asarray(t).flatten()
        u = 2 * np.pi * t
        c = 2 * np.pi
        out = np.zeros((len(t), 3))
        out[:, 0] =  scale * c * (np.cos(u) + 4 * np.cos(2 * u))
        out[:, 1] = -scale2 * c * (np.sin(u) - 4 * np.sin(2 * u))
        out[:, 2] = -scale3 * c * 3 * np.cos(3 * u)
        return out
    return Curve(F=trefoil_F, dF=trefoil_dF, ambient_dim=3)

def make_three_segments(scale):
    def three_segments_F(t, scale=1.0):
        t = np.asarray(t).flatten()
        seg = np.minimum((3 * t).astype(int), 2)
        s = 3 * t - seg
        angles = 2 * np.pi * seg / 3.0
        out = np.zeros((len(t), 3))
        out[:, 0] = scale * s * np.cos(angles)
        out[:, 1] = scale * s * np.sin(angles)
        return out

    def three_segments_dF(t):
        t = np.asarray(t).flatten()
        seg = np.minimum((3 * t).astype(int), 2)
        angles = 2 * np.pi * seg / 3.0
        out = np.zeros((len(t), 3))
        out[:, 0] = 3.0 * scale * np.cos(angles)
        out[:, 1] = 3.0 * scale * np.sin(angles)
        return out

    return Curve(F=three_segments_F, dF=three_segments_dF, ambient_dim=2)

def make_circle(scale):
    def circle_F(t):
        t = np.asarray(t).flatten()
        return np.column_stack([scale * np.cos(2*np.pi*t), scale * np.sin(2*np.pi*t)])


    def circle_dF(t):
        t = np.asarray(t).flatten()
        c = 2 * np.pi * scale
        return np.column_stack([-c * np.sin(2*np.pi*t), c * np.cos(2*np.pi*t)])

    return Curve(F=circle_F, dF=circle_dF, ambient_dim=2)

def make_spiral(scale, twists=3.0):
    def spiral_F(t):
        t=np.asarray(t).flatten()
        c = 2 * np.pi * twists
        return scale * t[:, None] * np.column_stack([np.cos(c * t), np.sin(c * t)])

    def spiral_dF(t):
        t=np.asarray(t).flatten()
        c=2*np.pi*twists
        dx = scale * (np.cos(c * t) - c * t * np.sin(c * t))
        dy = scale * (np.sin(c * t) + c * t * np.cos(c * t))
        return np.column_stack([dx, dy])


    return Curve(F=spiral_F, dF=spiral_dF, ambient_dim=2)

def make_nested_circles(radii=(0.1, 0.2, 0.3)):
    radii = np.asarray(radii, dtype=float)
    n_circles=len(radii)
    def nested_circles_F(t):
        t=np.asarray(t).flatten()
        idx=np.minimum((n_circles * t).astype(int), n_circles - 1)
        r=radii[idx]
        local_t=n_circles * t-idx
        return np.column_stack([r * np.cos(2 * np.pi * local_t), 
                                r * np.sin(2 * np.pi * local_t)])
    def nested_circles_dF(t):
        t = np.asarray(t).flatten()
        idx = np.minimum((n_circles * t).astype(int), n_circles - 1)
        local_t = n_circles * t - idx
        r = radii[idx]
        c = 2 * np.pi * n_circles
        return np.column_stack([-r * c * np.sin(2 * np.pi * local_t), 
                                 r * c * np.cos(2 * np.pi * local_t)])
    
    return Curve(F=nested_circles_F, dF=nested_circles_dF, ambient_dim=2)

def make_trefoil2(major_r=2.0, minor_r=1.0):
    def trefoil2_F(t):
        t = np.asarray(t).flatten()
        u = 2 * np.pi * t
        out = np.zeros((len(t), 3))
        
        # Torus knot equations: sweeps around the major radius while spiraling the minor radius
        out[:, 0] = (major_r + minor_r * np.cos(3 * u)) * np.cos(2 * u)
        out[:, 1] = (major_r + minor_r * np.cos(3 * u)) * np.sin(2 * u)
        out[:, 2] = minor_r * np.sin(3 * u)
        return out
    
    def trefoil2_dF(t):
        t = np.asarray(t).flatten()
        u = 2 * np.pi * t
        c = 2 * np.pi
        out = np.zeros((len(t), 3))
        
        # Product and chain rules for the derivatives
        dx_du = -3 * minor_r * np.sin(3 * u) * np.cos(2 * u) - 2 * (major_r + minor_r * np.cos(3 * u)) * np.sin(2 * u)
        dy_du = -3 * minor_r * np.sin(3 * u) * np.sin(2 * u) + 2 * (major_r + minor_r * np.cos(3 * u)) * np.cos(2 * u)
        dz_du =  3 * minor_r * np.cos(3 * u)
        
        out[:, 0] = c * dx_du
        out[:, 1] = c * dy_du
        out[:, 2] = c * dz_du
        return out
        
    return Curve(F=trefoil2_F, dF=trefoil2_dF, ambient_dim=3)

def make_cassini_oval(scale=1.0) -> Curve:
    """Cassini oval embedded in R3, generated as a 1D curve."""
    def cassini_F(t):
        t = np.asarray(t).flatten()
        theta = 2 * np.pi * t  # Map [0, 1] t-domain to [0, 2pi]
        out = np.zeros((len(t), 3))
        
        u = np.cos(2 * theta)
        v = np.sqrt(u**2 + 0.2)
        r = np.sqrt(u + v)
        
        out[:, 0] = scale * r * np.cos(theta)
        out[:, 1] = scale * r * np.sin(theta)
        out[:, 2] = scale * 0.3 * np.sin(theta + np.pi)
        return out

    def cassini_dF(t):
        t = np.asarray(t).flatten()
        theta = 2 * np.pi * t
        c = 2 * np.pi 
        out = np.zeros((len(t), 3))
        
        # Analytical derivatives for velocity/arc-length calculations
        u = np.cos(2 * theta)
        du_dtheta = -2 * np.sin(2 * theta)
        
        v = np.sqrt(u**2 + 0.2)
        dv_dtheta = (u / v) * du_dtheta
        
        r = np.sqrt(u + v)
        dr_dtheta = 0.5 * (1.0 / r) * (du_dtheta + dv_dtheta)
        
        dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
        dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
        dz_dtheta = 0.3 * np.cos(theta + np.pi)
        
        out[:, 0] = scale * c * dx_dtheta
        out[:, 1] = scale * c * dy_dtheta
        out[:, 2] = scale * c * dz_dtheta
        return out
        
    return Curve(F=cassini_F, dF=cassini_dF, ambient_dim=3)

def make_lissajous(amps=(1.0, 1.0), freqs=(2, 3)) -> Curve:
    """
    Constant-norm curve on a flat torus in R^(2m).
    Amplitudes should be roughly equal to perfectly condition the covariance.
    Frequencies should be coprime for a closed loop (knot).
    """
    def lissajous_F(t):
        t = np.asarray(t).flatten()
        out = np.zeros((len(t), 2 * len(amps)))
        for i, (a, w) in enumerate(zip(amps, freqs)):
            c = 2 * np.pi * w * t
            out[:, 2*i]   = a * np.cos(c)
            out[:, 2*i+1] = a * np.sin(c)
        return out

    def lissajous_dF(t):
        t = np.asarray(t).flatten()
        out = np.zeros((len(t), 2 * len(amps)))
        for i, (a, w) in enumerate(zip(amps, freqs)):
            c = 2 * np.pi * w
            arg = c * t
            out[:, 2*i]   = -a * c * np.sin(arg)
            out[:, 2*i+1] =  a * c * np.cos(arg)
        return out

    return Curve(F=lissajous_F, dF=lissajous_dF, ambient_dim=2 * len(amps))

