"""
Determine the 3D spin axis of a pitched baseball from Statcast/Trackman trajectory
data, following Alan M. Nathan's "Determining the 3D Spin Axis from Statcast Data":
https://baseball.physics.illinois.edu/trackman/spinaxis.pdf

Statcast/Trackman only measures the total spin rate and the trajectory (initial
velocity + average acceleration). The acceleration's "Magnus" component reveals the
transverse spin (the part of the spin that actually curves the pitch), but not the
gyrospin (the bullet-like spin along the direction of travel, which produces no
Magnus force). Combining the trajectory-derived transverse spin with Trackman's
measured total spin magnitude recovers the full 3D spin vector, up to a sign
ambiguity on the gyrospin component that is resolved from pitcher handedness.

Coordinate system (PITCHf/x / Statcast convention): origin at the back tip of home
plate, +x toward the catcher's right, +y toward the pitcher, +z up.
"""
import numpy as np

GRAVITY_FTS2 = 32.174
Y_MEASUREMENT = 50.0          # ft; Statcast reports vx0/vy0/vz0 at this y
Y_PLATE = 17 / 12             # ft; front edge of home plate
Y_RUBBER = 60.5               # ft; front edge of the pitching rubber

A_FIT, B_FIT = 0.336, 6.041   # C_L = A[1 - exp(-B*S)] fit constants, Eq. 8 (May 2020)
BALL_RADIUS_FT = 1.45 / 12    # official ball circumference ~9.125in -> radius
BALL_MASS_SLUG = (5.125 / 16) / GRAVITY_FTS2  # official ball weight ~5.125oz


def air_density(temperature_f=70.0, elevation_ft=0.0, humidity_pct=50.0):
    """Air density (slug/ft^3) from temperature, elevation, and relative humidity."""
    pressure_inhg = 29.92 * (1 - 6.8753e-6 * elevation_ft) ** 5.2559
    temp_c = (temperature_f - 32) * 5.0 / 9.0
    temp_r = temperature_f + 459.67

    sat_vapor_inhg = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3)) * 0.02953
    vapor_pressure_inhg = sat_vapor_inhg * (humidity_pct / 100.0)
    dry_pressure_inhg = pressure_inhg - vapor_pressure_inhg

    inhg_to_psf = 70.726
    r_dry, r_vapor = 1716.0, 2760.5
    return (dry_pressure_inhg * inhg_to_psf) / (r_dry * temp_r) + \
           (vapor_pressure_inhg * inhg_to_psf) / (r_vapor * temp_r)


def drag_constant_K(temperature_f=70.0, elevation_ft=0.0, humidity_pct=50.0):
    """K = (1/2)*rho*A/m (ft^-1), the constant in the drag/Magnus acceleration formulas."""
    rho = air_density(temperature_f, elevation_ft, humidity_pct)
    ball_area_ft2 = np.pi * BALL_RADIUS_FT ** 2
    return 0.5 * rho * ball_area_ft2 / BALL_MASS_SLUG


def compute_3d_spin_axis(df, K=None, temperature_f=70.0, elevation_ft=0.0, humidity_pct=50.0):
    """
    Compute the 3D spin axis of pitched baseballs from Statcast trajectory data.

    Works on a DataFrame of many pitches or a single pitch (dict/Series) since it
    only ever indexes its input by column name.

    Required fields: vx0, vy0, vz0, ax, ay, az (9P fit velocity/acceleration,
    Statcast convention), release_extension (ft), release_spin_rate (rpm).
    Optional: p_throws ('R'/'L') to resolve the gyrospin sign -- gyrospin is assumed
    parallel to the velocity for RHP and antiparallel for LHP, per the paper's guess
    (Sec. II.A); defaults to the RHP convention if omitted.

    Returns a dict with:
      lift_coefficient      C_L, Eq. 7
      transverse_spin_rpm   magnitude of the trajectory-derived "useful" spin, Eq. 9
      gyro_spin_rpm         signed magnitude of the bullet/gyro spin, Eq. 4
      spin_axis_{x,y,z}_rpm the full 3D spin vector, Eq. 11
      spin_efficiency       transverse_spin / total_spin (NaN if >1, a physical
                             impossibility left blank per the paper's Sec. II.F)
      gyro_angle_deg        theta: 0 deg = pure transverse spin, 90 deg = pure gyroball
      spin_direction_deg    phi: azimuth of the spin axis in the x-z plane (catcher's
                             view), measured from +x, 0-360 deg
    """
    if K is None:
        K = drag_constant_K(temperature_f, elevation_ft, humidity_pct)

    vx0, vy0, vz0 = df['vx0'], df['vy0'], df['vz0']
    ax, ay, az = df['ax'], df['ay'], df['az']
    release_y = Y_RUBBER - df['release_extension']

    # Velocity at release, worked backward from the y=50ft measurement point,
    # and at the front of home plate, using v(y)^2 = v0^2 + 2*a*(y-y0).
    vy_release = -np.sqrt(vy0 ** 2 + 2 * ay * (release_y - Y_MEASUREMENT))
    t_release_to_50 = (vy0 - vy_release) / ay
    vx_release = vx0 - ax * t_release_to_50
    vz_release = vz0 - az * t_release_to_50

    vy_plate = -np.sqrt(vy0 ** 2 - 2 * ay * (Y_MEASUREMENT - Y_PLATE))
    t_50_to_plate = (vy_plate - vy0) / ay
    vx_plate = vx0 + ax * t_50_to_plate
    vz_plate = vz0 + az * t_50_to_plate

    # Mean velocity over the flight, exact for constant acceleration (avg of endpoints).
    vbar_x = 0.5 * (vx_release + vx_plate)
    vbar_y = 0.5 * (vy_release + vy_plate)
    vbar_z = 0.5 * (vz_release + vz_plate)
    vbar_mag = np.sqrt(vbar_x ** 2 + vbar_y ** 2 + vbar_z ** 2)
    vhat_x, vhat_y, vhat_z = vbar_x / vbar_mag, vbar_y / vbar_mag, vbar_z / vbar_mag

    # Strip gravity, then strip drag -- the component of a* along vhat -- to isolate
    # the Magnus acceleration (Eq. 6), leaving the part of a* perpendicular to vhat.
    astar_x, astar_y, astar_z = ax, ay, az + GRAVITY_FTS2
    astar_dot_vhat = astar_x * vhat_x + astar_y * vhat_y + astar_z * vhat_z
    aM_x = astar_x - astar_dot_vhat * vhat_x
    aM_y = astar_y - astar_dot_vhat * vhat_y
    aM_z = astar_z - astar_dot_vhat * vhat_z
    aM_mag = np.sqrt(aM_x ** 2 + aM_y ** 2 + aM_z ** 2)
    aMhat_x, aMhat_y, aMhat_z = aM_x / aM_mag, aM_y / aM_mag, aM_z / aM_mag

    # Lift coefficient (Eq. 7) inverted to the transverse spin magnitude (Eq. 9).
    # CL is bounded above by A_FIT (Eq. 8's asymptote); above that the fit can't be
    # inverted, so treat it as missing rather than evaluating log of a non-positive number.
    CL = aM_mag / (K * vbar_mag ** 2)
    CL_invertible = np.where(CL < A_FIT, CL, np.nan)
    transverse_spin = (vbar_mag / (BALL_RADIUS_FT * B_FIT)) * np.log(A_FIT / (A_FIT - CL_invertible))

    # Transverse spin direction, Eq. 10: what = vhat x aMhat.
    wThat_x = vhat_y * aMhat_z - vhat_z * aMhat_y
    wThat_y = vhat_z * aMhat_x - vhat_x * aMhat_z
    wThat_z = vhat_x * aMhat_y - vhat_y * aMhat_x
    wT_x, wT_y, wT_z = transverse_spin * wThat_x, transverse_spin * wThat_y, transverse_spin * wThat_z

    # Gyrospin magnitude from total spin (Trackman) and transverse spin, Eq. 4.
    # Its direction is +-vhat; resolved via pitcher handedness (Sec. II.A).
    omega_total = df['release_spin_rate'] * 2 * np.pi / 60.0
    diff = omega_total ** 2 - transverse_spin ** 2
    gyro_spin_mag = np.sqrt(np.where(diff >= 0, diff, np.nan))

    p_throws = df['p_throws'] if 'p_throws' in df else 'R'
    gyro_sign = np.where(np.asarray(p_throws) == 'L', -1, 1)

    wG_x, wG_y, wG_z = gyro_sign * gyro_spin_mag * vhat_x, \
        gyro_sign * gyro_spin_mag * vhat_y, gyro_sign * gyro_spin_mag * vhat_z

    wx, wy, wz = wT_x + wG_x, wT_y + wG_y, wT_z + wG_z

    rad_s_to_rpm = 60 / (2 * np.pi)
    theta = np.degrees(np.arctan2(wy, np.sqrt(wx ** 2 + wz ** 2)))
    phi = np.degrees(np.arctan2(wz, wx)) % 360

    return {
        'lift_coefficient': CL,
        'transverse_spin_rpm': transverse_spin * rad_s_to_rpm,
        'gyro_spin_rpm': gyro_sign * gyro_spin_mag * rad_s_to_rpm,
        'spin_axis_x_rpm': wx * rad_s_to_rpm,
        'spin_axis_y_rpm': wy * rad_s_to_rpm,
        'spin_axis_z_rpm': wz * rad_s_to_rpm,
        'spin_efficiency': transverse_spin / omega_total,
        'gyro_angle_deg': theta,
        'spin_direction_deg': phi,
    }
