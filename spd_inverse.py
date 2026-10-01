# -*- coding: utf-8 -*-
"""
Regularised inverse of the kernel-renormalisation matrix E.

E_i = Sum_j V_j (x_j - x_i) (x) grad W_ij  is *symmetric positive semi-definite by
construction*: grad W_ij is parallel to (x_i - x_j), so every term is a positive
multiple of dx (x) dx.  Its spectrum is therefore a direct, interpretable measure of
how well the neighbourhood of particle i resolves each spatial direction:

    e_k ~ 1     direction k is fully populated (interior of the body)
    e_k -> 0    direction k is starved -- a free surface (normal direction), a
                crack tip, a single-particle ligament, or neighbours that happen to
                be collinear (2D) / coplanar (3D)

A plain `E.inverse()` amplifies exactly those starved directions by 1/e_k, i.e. it
applies the largest correction where there is the least information to correct with,
and it is unbounded as e_k -> 0.

The regularisation here is *direction-wise fallback to the identity*, not a scalar
clamp: each eigendirection is corrected by 1/e_k where it is well resolved, and left
uncorrected (weight 1) where it is not, with a smooth blend in between so the operator
is continuous as particles move.

    w(e) = beta/max(e, tol) + (1 - beta),   beta = smoothstep((e - tol)/tol)
    L    = Sum_k w(e_k) n_k (x) n_k

    e >= 2*tol  ->  w = 1/e     (full correction)
    e <= tol    ->  w = 1       (no correction: fall back to the bare gradient)

Two properties that make this safe to put in front of a force or a strain rate:

  * bounded by 1/tol, with the bound chosen explicitly rather than by accident;
  * the worst case is the *uncorrected* operator, which is what the scheme would
    have used anyway -- degeneracy degrades the correction, it cannot blow it up.

Note that `tol` should differ per consumer.  The surface normal <grad lambda> *wants*
the amplification in the starved direction (lambda varies mainly along the surface
normal, which is precisely the deficient direction), so it uses a tolerance that only
catches genuine degeneracy.  A velocity gradient feeding a constitutive law wants a
much more conservative tolerance.
"""
import taichi as ti


@ti.func
def _weight(e: ti.f32, tol: ti.f32) -> ti.f32:
    """Per-eigendirection weight: 1/e where well resolved, 1 where starved."""
    u = ti.min(1.0, ti.max(0.0, (e - tol) / tol))
    beta = u * u * (3.0 - 2.0 * u)
    return beta / ti.max(e, tol) + (1.0 - beta)


@ti.func
def spd_inverse_2x2(E, tol: ti.f32):
    """
    Regularised inverse of a symmetric positive semi-definite 2x2 matrix.

    Returns the 2x2 matrix L used as  grad f = (bare gradient sum) @ L.
    """
    a = E[0, 0]
    b = 0.5 * (E[0, 1] + E[1, 0])
    d = E[1, 1]

    disc = ti.sqrt(ti.max((a - d) * (a - d) + 4.0 * b * b, 0.0))
    e1 = 0.5 * (a + d + disc)          # larger
    e2 = 0.5 * (a + d - disc)          # smaller

    # Eigenvector of e1.  Both (b, e1-a) and (e1-d, b) span it; take the longer one,
    # which is the numerically stable choice.  If E is isotropic (disc == 0) any
    # orthonormal pair works, so fall back to the axes.
    n1 = ti.Vector([1.0, 0.0])
    scale = ti.abs(a) + ti.abs(d) + 1e-30
    if disc > 1e-9 * scale:
        v1 = ti.Vector([b, e1 - a])
        v2 = ti.Vector([e1 - d, b])
        if v1.norm() >= v2.norm():
            n1 = v1 / v1.norm()
        else:
            n1 = v2 / v2.norm()
    n2 = ti.Vector([-n1[1], n1[0]])

    return (_weight(e1, tol) * n1.outer_product(n1) +
            _weight(e2, tol) * n2.outer_product(n2))


@ti.func
def spd_inverse_3x3(E, tol: ti.f32):
    """
    Regularised inverse of a symmetric positive semi-definite 3x3 matrix.

    E is symmetrised before the eigendecomposition: it is symmetric analytically, but
    the neighbour sum accumulates it in float32 and ti.sym_eig assumes exact symmetry.
    """
    Es = 0.5 * (E + E.transpose())
    evals, evecs = ti.sym_eig(Es, ti.f32)

    L = ti.Matrix.zero(ti.f32, 3, 3)
    for k in ti.static(range(3)):
        n = ti.Vector([evecs[0, k], evecs[1, k], evecs[2, k]])
        L += _weight(evals[k], tol) * n.outer_product(n)
    return L


@ti.func
def min_eig_vec_2x2(E):
    """
    Unit eigenvector of the smallest eigenvalue of a symmetric 2x2 matrix.

    E = Sum_j V_j (x_j - x_i) (x) grad W_ij is positive semi-definite, and its smallest
    eigenvalue belongs to the direction the neighbourhood resolves worst -- at a free
    surface, the surface normal.  Colle et al. (2019) use exactly this as the normal for
    the tangential projection of the shift, which costs nothing here because the
    eigendecomposition is already being done for lambda.

    The sign is arbitrary (an eigenvector is defined up to it); the projection
    dr - (dr.n)n does not depend on it.  Returns a 2-vector.
    """
    a = E[0, 0]
    b = 0.5 * (E[0, 1] + E[1, 0])
    d = E[1, 1]

    disc = ti.sqrt(ti.max((a - d) * (a - d) + 4.0 * b * b, 0.0))
    e1 = 0.5 * (a + d + disc)          # larger

    # Same construction as spd_inverse_2x2: build the eigenvector of the LARGER
    # eigenvalue, where the two candidate spanning vectors are well separated, and
    # rotate by 90 degrees.  Doing it directly on the smaller one is ill-conditioned
    # exactly in the starved case this is wanted for.
    n1 = ti.Vector([1.0, 0.0])
    scale = ti.abs(a) + ti.abs(d) + 1e-30
    if disc > 1e-9 * scale:
        v1 = ti.Vector([b, e1 - a])
        v2 = ti.Vector([e1 - d, b])
        if v1.norm() >= v2.norm():
            n1 = v1 / v1.norm()
        else:
            n1 = v2 / v2.norm()
    return ti.Vector([-n1[1], n1[0]])


@ti.func
def min_eig_vec_3x3(E):
    """
    Unit eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix.

    E is symmetrised first, for the same reason as in spd_inverse_3x3: it is symmetric
    analytically but accumulated in float32, and ti.sym_eig assumes exact symmetry.
    """
    Es = 0.5 * (E + E.transpose())
    evals, evecs = ti.sym_eig(Es, ti.f32)

    k_min = 0
    if evals[1] < evals[k_min]:
        k_min = 1
    if evals[2] < evals[k_min]:
        k_min = 2
    return ti.Vector([evecs[0, k_min], evecs[1, k_min], evecs[2, k_min]])
