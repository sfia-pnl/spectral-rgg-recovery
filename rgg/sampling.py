import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
import rgg.curves as curves


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

def generate_manifold_samples(n, curve, n_grid=1000):
    """Sample n points uniformly w.r.t. arc length from a 1D curve."""
    t_grid = np.linspace(0, 1, n_grid)
    speed_grid = curve.speed(t_grid)
    S = _trapz(speed_grid, t_grid)
    cdf = cumulative_trapezoid(speed_grid, t_grid, initial=0) / S
    inverse_CDF = CubicSpline(cdf, t_grid)
    u_samples = np.random.rand(n)
    t_samples = inverse_CDF(u_samples)
    points = curve.F(t_samples)
    return points, t_samples

def sample_swiss_roll(n,  params, signal_dim=3):
    """Sample n points from a swiss roll uniformly wrt to the surface area, not a 1D curve"""
    twists, scale, height=params['twists'], params['scale'], params['height']
    t = np.random.uniform(0.0, 1.0, n)
    u = np.random.uniform(0.0, 1.0, n)
    angle = np.pi + 2 * np.pi * twists * t
    radius = 0.3 + t
    out = np.zeros((n, signal_dim))
    out[:, 0] = scale * radius * np.cos(angle)
    out[:, 1] = scale * u * height          
    out[:, 2] = scale * radius * np.sin(angle)
    latent = np.column_stack([t, u])
    return out, latent

def sample_sphere(n, signal_dim=3):
    """Sample n points uniformly on the unit sphere S^{signal_dim - 1}."""
    Z = np.random.randn(n, signal_dim)
    X = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    theta = np.arccos(X[:, 2])
    return X, theta

def sample_nested_spheres(n, params, signal_dim=3):
    """Sample n points from multiple concentric spheres."""
    # Default to two spheres of radius 1.0 and 2.0 if not provided
    radii = params.get('radii', (1.0, 2.0))
    
    # Sample uniformly on the unit sphere
    Z = np.random.randn(n, signal_dim)
    X = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    
    # Randomly assign each point to one of the radii with equal probability
    choices = np.random.choice(radii, size=(n, 1))
    X_nested = X * choices
    
    # Use the assigned radius as the coloring
    latent = choices.flatten()
    
    return X_nested, latent

def sample_torus(n: int, R=2.0, r_minor=0.8):
    """
    Uniformly samples n points from the surface of a torus in R3.
    Rejection sampling ensures strict surface uniformity.
    """
    u_samples = []
    max_val = R + r_minor
    while len(u_samples) < n:
        u_cand = np.random.uniform(0, 2 * np.pi, size=n)
        y = np.random.uniform(0, max_val, size=n)
        # Accept proportional to the volume element (R + r*cos(u))
        accept = y <= (R + r_minor * np.cos(u_cand))
        u_samples.extend(u_cand[accept])
    
    u = np.array(u_samples[:n])
    v = np.random.uniform(0, 2 * np.pi, size=n)
    
    X = np.zeros((n, 3))
    X[:, 0] = (R + r_minor * np.cos(u)) * np.cos(v)
    X[:, 1] = (R + r_minor * np.cos(u)) * np.sin(v)
    X[:, 2] = r_minor * np.sin(u)
    
    latent = np.column_stack([u, v])
    return X, latent

def sample_smiley_face(n: int, scale=1.0):
    """
    Uniformly samples n points from the 2D smiley face sets (S1 ∪ S2 ∪ S3 ∪ S4).
    Uses vectorised rejection sampling over the tight bounding box.
    """
    points = []
    labels = [] 
    
    while len(points) < n:
        batch_size = n * 5
        xy = np.random.uniform(-1.5, 1.5, size=(batch_size, 2))
        x, y = xy[:, 0], xy[:, 1]
        r2 = x**2 + y**2
        
        m1 = (x - 0.5)**2 + (y - 0.5)**2 <= 0.1      # Right eye
        m2 = (x + 0.5)**2 + (y - 0.5)**2 <= 0.1      # Left eye
        m3 = (1.8 <= r2) & (r2 <= 2.0)               # Outer boundary
        m4 = (0.9 <= r2) & (r2 <= 1.1) & (y <= 0)    # Mouth
        
        valid_mask = m1 | m2 | m3 | m4
        if np.any(valid_mask):
            valid_pts = xy[valid_mask]
            
            # Map points to discrete components for the 'color_by' visualizer
            c = np.zeros(len(valid_pts))
            vx, vy = valid_pts[:, 0], valid_pts[:, 1]
            vr2 = vx**2 + vy**2
            c[(vx - 0.5)**2 + (vy - 0.5)**2 <= 0.1] = 1
            c[(vx + 0.5)**2 + (vy - 0.5)**2 <= 0.1] = 2
            c[(1.8 <= vr2) & (vr2 <= 2.0)] = 3
            c[(0.9 <= vr2) & (vr2 <= 1.1) & (vy <= 0)] = 4
            
            points.extend(valid_pts)
            labels.extend(c)
            
    X = np.array(points[:n]) * scale
    latent = np.array(labels[:n])
    
    return X, latent

def sample_clifford_torus(n: int, R1=1.0, R2=1.0):
    """
    Uniformly samples n points from a Clifford Torus embedded in R4.
    If R1 == R2, every point sits at exact constant distance from the origin.
    """
    u = np.random.uniform(0, 2 * np.pi, n)
    v = np.random.uniform(0, 2 * np.pi, n)
    
    X = np.column_stack([
        R1 * np.cos(u),
        R1 * np.sin(u),
        R2 * np.cos(v),
        R2 * np.sin(v)
    ])
    
    latent = np.column_stack([u, v])
    return X, latent

def sample_ellipsoid(n: int, radii=(1.0, 1.0, 0.9)):
    """
    Uniformly samples n points from an ellipsoid surface in R^d using rejection sampling.
    Used to test the degradation of the algorithm as eccentricity increases.
    """
    radii = np.asarray(radii, dtype=float)
    dim = len(radii)
    
    samples = []
    # The maximum weight for the surface area element occurs at the shortest radius
    max_w = 1.0 / np.min(radii) 
    
    while len(samples) < n:
        batch_size = n * 2
        # Generate uniform on unit sphere
        g = np.random.randn(batch_size, dim)
        u = g / np.linalg.norm(g, axis=1, keepdims=True)
        
        # Stretch to ellipsoid
        x = u * radii
        
        # Compute exact surface normals to adjust for uniform density
        normals = x / (radii**2)
        weights = np.linalg.norm(normals, axis=1)
        
        # Accept proportional to the surface area element
        accept = np.random.rand(batch_size) < (weights / max_w)
        samples.extend(x[accept])
        
    X = np.array(samples[:n])
    latent = X[:, 0]  # Color by the primary axis
    return X, latent
