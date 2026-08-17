#!/usr/bin/env python3
import numpy as np
import random
import unittest

import opendbc.safety.tests.common as common
# BluePilot: MAX_LATERAL_ACCEL here is the CAN FD limit from the carcontroller (~2.4 m/s^2,
# ISO minus the roll term), which is what ford.h's limit_lateral_acceleration path enforces.
# Upstream's lateral.MAX_LATERAL_ACCEL/MAX_LATERAL_JERK (~3.6, ISO plus roll) belong to the
# ISO-only curvature path BluePilot does not use -- ford.h sets use_rate_lookup instead -- so they
# are deliberately not imported here. See CURVATURE_RATE_LOOKUP_* below.
from opendbc.car.ford.carcontroller import MAX_LATERAL_ACCEL
from opendbc.car.ford.values import CAR, FordFlags, FordSafetyFlags
from opendbc.car.interfaces import scale_tire_stiffness
from opendbc.car.vehicle_model import VehicleModel, calc_slip_factor
from opendbc.sunnypilot.car.ford.values_ext import FORD_PINION_GEOMETRY_INDEX, FORD_PINION_GEOMETRY_SHIFT, FordSafetyFlagsSP
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety

MSG_BrakeSysFeatures = 0x415       # RX from ABS, for vehicle speed
MSG_EngVehicleSpThrottle2 = 0x202  # RX from PCM, for second vehicle speed
MSG_Yaw_Data_FD1 = 0x91            # RX from RCM, for yaw rate
MSG_Steering_Data_FD1 = 0x083      # TX by OP, various driver switches and LKAS/CC buttons
MSG_ACCDATA = 0x186                # TX by OP, ACC controls
MSG_ACCDATA_3 = 0x18A              # TX by OP, ACC/TJA user interface
MSG_Lane_Assist_Data1 = 0x3CA      # TX by OP, Lane Keep Assist
MSG_LateralMotionControl = 0x3D3   # TX by OP, Lateral Control message
MSG_LateralMotionControl2 = 0x3D6  # TX by OP, alternate Lateral Control message
MSG_IPMA_Data = 0x3D8              # TX by OP, IPMA and LKAS user interface


def checksum(msg):
  addr, dat, bus = msg
  ret = bytearray(dat)

  if addr == MSG_Yaw_Data_FD1:
    chksum = dat[0] + dat[1]  # VehRol_W_Actl
    chksum += dat[2] + dat[3]  # VehYaw_W_Actl
    chksum += dat[5]  # VehRollYaw_No_Cnt
    chksum += dat[6] >> 6  # VehRolWActl_D_Qf
    chksum += (dat[6] >> 4) & 0x3  # VehYawWActl_D_Qf
    chksum = 0xff - (chksum & 0xff)
    ret[4] = chksum

  elif addr == MSG_BrakeSysFeatures:
    chksum = dat[0] + dat[1]  # Veh_V_ActlBrk
    chksum += (dat[2] >> 2) & 0xf  # VehVActlBrk_No_Cnt
    chksum += dat[2] >> 6  # VehVActlBrk_D_Qf
    chksum = 0xff - (chksum & 0xff)
    ret[3] = chksum

  elif addr == MSG_EngVehicleSpThrottle2:
    chksum = (dat[2] >> 3) & 0xf  # VehVActlEng_No_Cnt
    chksum += (dat[4] >> 5) & 0x3  # VehVActlEng_D_Qf
    chksum += dat[6] + dat[7]  # Veh_V_ActlEng
    chksum = 0xff - (chksum & 0xff)
    ret[1] = chksum

  return addr, ret, bus


class Buttons:
  CANCEL = 0
  RESUME = 1
  TJA_TOGGLE = 2


# Ford safety has four different configurations tested here:
#  * CAN with openpilot longitudinal
#  * CAN FD with stock longitudinal
#  * CAN FD with openpilot longitudinal

class TestFordSafetyBase(common.CarSafetyTest):
  # BluePilot: sunnypilot SP safety param (current_safety_param_sp), set before
  # set_safety_hooks in every concrete setUp -- ford_init reads it. 0 = stock behavior;
  # the pinion-curvature classes below override it (Toyota SAFETY_PARAM_SP convention).
  SAFETY_PARAM_SP: int = 0

  STANDSTILL_THRESHOLD = 1
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                                 MSG_LateralMotionControl2, MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                               MSG_LateralMotionControl2, MSG_IPMA_Data]}

  STEER_MESSAGE = 0

  # Curvature control limits
  DEG_TO_CAN = 50000   # CAN units per rad/m
  MAX_CURVATURE = 0.02 # rad/m, 1000 CAN units
  MAX_CURVATURE_ERROR = 0.002         # rad/m, 100 CAN units
  CURVATURE_ERROR_MIN_SPEED = 10.0    # m/s
  LATERAL_FREQUENCY = 20              # Hz, message rate

  # BluePilot: ford.h sets use_rate_lookup, so the per-frame curvature rate limit comes from these
  # measured tables (curvature_rate_up/down_lookup in the FORD_LIMITS macro), NOT from upstream's
  # ISO lateral-jerk formula. The two differ enormously at low speed -- at 2 m/s the ISO formula
  # permits ~0.18 rad/m of curvature change per frame while the table permits 0.0025 -- so the
  # ISO-derived expectations these helpers used to carry demanded that safety allow steps it
  # rightly blocks. Up and down tables are identical in ford.h, so one copy is enough.
  CURVATURE_RATE_LOOKUP_BP = (5., 16., 25.)
  CURVATURE_RATE_LOOKUP_V = (0.0025, 0.0014, 0.00018)
  # safety_interpolate() runs in float32 and the speed the safety sees is the DBC-quantized one,
  # so a boundary computed here can land one CAN unit either side of the safety's. Boundary probes
  # below step out by this much before asserting, which still pins each limit to within 2 CAN units
  # (4e-5 rad/m).
  RATE_LIMIT_TOL_CAN = 1

  cnt_speed = 0
  cnt_speed_2 = 0
  cnt_yaw_rate = 0
  cnt_lat_ctl = 0

  packer: CANPackerSafety
  safety: libsafety_py.LibSafety

  # BluePilot: retained from pre-sync. ford.h applies the lateral-accel cap only where
  # limit_lateral_acceleration is set (CAN FD / Q4), not on CAN / Q3 as upstream's ISO-only
  # curvature path does, so every lateral-accel expectation stays bus-dependent.
  # See FORD_LIMITS in ford.h -- and the report note on the missing Q3 cap.
  @property
  def limit_lateral_accel(self):
    return self.STEER_MESSAGE == MSG_LateralMotionControl2

  def _max_curvature_allowed_can(self, speed):
    """The largest curvature safety will accept at this speed, in CAN units."""
    max_curvature_can = round(self.MAX_CURVATURE * self.DEG_TO_CAN)
    if not self.limit_lateral_accel:
      return max_curvature_can
    return min(self._get_max_curvature_can(speed), max_curvature_can)

  def _get_max_curvature_can(self, speed):
    fudged_speed = max(speed - 1.0, 1.0)
    return int(MAX_LATERAL_ACCEL / (fudged_speed * fudged_speed) * self.DEG_TO_CAN) + 1

  def _curvature_rate_lookup(self, speed):
    # mirrors safety_interpolate() in helpers.h: clamped piecewise-linear interp, float32 like the C
    return float(np.interp(np.float32(speed), self.CURVATURE_RATE_LOOKUP_BP, self.CURVATURE_RATE_LOOKUP_V))

  def _get_max_curvature_delta_can(self, speed):
    # safety fudges the speed down by 1 m/s so its rate limit sits slightly above openpilot's
    return int(self._curvature_rate_lookup(speed - 1.0) * self.DEG_TO_CAN) + 1

  def _get_max_curvature_delta_relaxed_can(self, speed):
    # flipped fudge, this is the least movement toward the error bounds safety requires
    return int(self._curvature_rate_lookup(speed + 1.0) * self.DEG_TO_CAN) - 1

  def _get_max_curvature_relaxed_can(self, speed):
    # The most safety can ever require the command to move toward measured. BluePilot's
    # use_rate_lookup branch clamps the required movement to the curvature signal range only
    # (SAFETY_CLAMP to +/-max_curvature in lateral.h); unlike upstream's ISO branch it does NOT
    # additionally clamp to the lateral acceleration cap, so above that cap the requirement and the
    # cap can disagree. That divergence is reported separately -- it is a false-positive (blocked
    # message) risk, never a permissive one, so this test asserts the implemented bound.
    del speed
    return round(self.MAX_CURVATURE * self.DEG_TO_CAN)

  def _set_prev_desired_angle(self, t):
    t = round(t * self.DEG_TO_CAN)
    self.safety.set_desired_curvature_last(t)

  def _reset_curvature_measurement(self, curvature, speed):
    for _ in range(6):
      self._rx(self._speed_msg(speed))
      self._rx(self._speed_msg_2(speed))
      self._rx(self._yaw_rate_msg(curvature, speed))

  def _meas_tol_can(self, speed):
    # CAN-unit uncertainty between the curvature _reset_curvature_measurement was asked for and the
    # one safety ends up holding. Zero on the yaw path; the pinion path quantizes to 0.1 deg.
    del speed
    return 0

  # Driver brake pedal
  def _user_brake_msg(self, brake: bool):
    # brake pedal and cruise state share same message, so we have to send
    # the other signal too
    enable = self.safety.get_controls_allowed()
    values = {
      "BpedDrvAppl_D_Actl": 2 if brake else 1,
      "CcStat_D_Actl": 5 if enable else 0,
    }
    return self.packer.make_can_msg_safety("EngBrakeData", 0, values)

  # ABS vehicle speed
  def _speed_msg(self, speed: float, quality_flag=True):
    values = {"Veh_V_ActlBrk": speed * 3.6, "VehVActlBrk_D_Qf": 3 if quality_flag else 0, "VehVActlBrk_No_Cnt": self.cnt_speed % 16}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("BrakeSysFeatures", 0, values, fix_checksum=checksum)

  # PCM vehicle speed
  def _speed_msg_2(self, speed: float, quality_flag=True):
    # Ford relies on speed for driver curvature limiting, so it checks two sources
    values = {"Veh_V_ActlEng": speed * 3.6, "VehVActlEng_D_Qf": 3 if quality_flag else 0, "VehVActlEng_No_Cnt": self.cnt_speed_2 % 16}
    self.__class__.cnt_speed_2 += 1
    return self.packer.make_can_msg_safety("EngVehicleSpThrottle2", 0, values, fix_checksum=checksum)

  # Standstill state
  def _vehicle_moving_msg(self, speed: float):
    values = {"VehStop_D_Stat": 1 if speed <= self.STANDSTILL_THRESHOLD else random.choice((0, 2, 3))}
    return self.packer.make_can_msg_safety("DesiredTorqBrk", 0, values)

  # Current curvature
  def _yaw_rate_msg(self, curvature: float, speed: float, quality_flag=True):
    values = {"VehYaw_W_Actl": curvature * speed, "VehYawWActl_D_Qf": 3 if quality_flag else 0,
              "VehRollYaw_No_Cnt": self.cnt_yaw_rate % 256}
    self.__class__.cnt_yaw_rate += 1
    return self.packer.make_can_msg_safety("Yaw_Data_FD1", 0, values, fix_checksum=checksum)

  # Drive throttle input
  def _user_gas_msg(self, gas: float):
    values = {"ApedPos_Pc_ActlArb": gas}
    return self.packer.make_can_msg_safety("EngVehicleSpThrottle", 0, values)

  # Cruise status
  def _pcm_status_msg(self, enable: bool):
    # brake pedal and cruise state share same message, so we have to send
    # the other signal too
    brake = self.safety.get_brake_pressed_prev()
    values = {
      "BpedDrvAppl_D_Actl": 2 if brake else 1,
      "CcStat_D_Actl": 5 if enable else 0,
    }
    return self.packer.make_can_msg_safety("EngBrakeData", 0, values)

  # LKAS command
  def _lkas_command_msg(self, action: int):
    values = {
      "LkaActvStats_D2_Req": action,
    }
    return self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, values)

  # BluePilot: angle_mode_engaged + shadow_curvature, packed into Lane_Assist_Data1's unused bits
  # (byte4 bit0, bytes 5-6 -- see fordcan_ext.py's create_lka_msg / ford.h's FORD_Lane_Assist_Data1
  # tx_hook check). Sent by openpilot itself, so this goes through _tx, not _rx.
  def _lka_bp_status_msg(self, angle_mode_engaged: bool, shadow_curvature: float, action: int = 0):
    values = {"LkaActvStats_D2_Req": action}
    addr, dat, bus = self.packer.make_can_msg("Lane_Assist_Data1", 0, values)
    dat = bytearray(dat)
    shadow_curvature_raw = int(round(shadow_curvature / 1e-6))
    shadow_curvature_raw = max(-32768, min(32767, shadow_curvature_raw)) & 0xFFFF
    dat[4] |= 1 if angle_mode_engaged else 0
    dat[5] = (shadow_curvature_raw >> 8) & 0xFF
    dat[6] = shadow_curvature_raw & 0xFF
    return libsafety_py.make_CANPacket(addr, bus, bytes(dat))

  # LCA command
  def _lat_ctl_msg(self, enabled: bool, path_offset: float, path_angle: float, curvature: float, curvature_rate: float,
                   increment_timer: bool = True):
    if increment_timer:
      self.safety.set_timer(self.cnt_lat_ctl * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_lat_ctl += 1
    if self.STEER_MESSAGE == MSG_LateralMotionControl:
      values = {
        "LatCtl_D_Rq": 1 if enabled else 0,
        "LatCtlPathOffst_L_Actl": path_offset,     # Path offset [-5.12|5.11] meter
        "LatCtlPath_An_Actl": path_angle,          # Path angle [-0.5|0.5235] radians
        "LatCtlCurv_NoRate_Actl": curvature_rate,  # Curvature rate [-0.001024|0.00102375] 1/meter^2
        "LatCtlCurv_No_Actl": curvature,           # Curvature [-0.02|0.02094] 1/meter
      }
      return self.packer.make_can_msg_safety("LateralMotionControl", 0, values)
    elif self.STEER_MESSAGE == MSG_LateralMotionControl2:
      values = {
        "LatCtl_D2_Rq": 1 if enabled else 0,
        "LatCtlPathOffst_L_Actl": path_offset,     # Path offset [-5.12|5.11] meter
        "LatCtlPath_An_Actl": path_angle,          # Path angle [-0.5|0.5235] radians
        "LatCtlCrv_NoRate2_Actl": curvature_rate,  # Curvature rate [-0.001024|0.001023] 1/meter^2
        "LatCtlCurv_No_Actl": curvature,           # Curvature [-0.02|0.02094] 1/meter
      }
      return self.packer.make_can_msg_safety("LateralMotionControl2", 0, values)

  # Cruise control buttons
  def _acc_button_msg(self, button: int, bus: int):
    values = {
      "CcAslButtnCnclPress": 1 if button == Buttons.CANCEL else 0,
      "CcAsllButtnResPress": 1 if button == Buttons.RESUME else 0,
      "TjaButtnOnOffPress": 1 if button == Buttons.TJA_TOGGLE else 0,
    }
    return self.packer.make_can_msg_safety("Steering_Data_FD1", bus, values)

  def test_rx_hook_speed_mismatch(self):
    for speed in np.arange(0, 40, 0.5):
      for speed_delta in np.arange(-5, 5, 0.1):
        speed_2 = round(max(speed + speed_delta, 0), 1)
        self._rx(self._speed_msg(speed))
        self._rx(self._speed_msg_2(speed_2))
        self.safety.set_controls_allowed(True)
        self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0))

        within_delta = abs(speed - speed_2) <= common.MAX_SPEED_DELTA
        self.assertEqual(self.safety.get_controls_allowed(), within_delta)

  def test_rx_hook(self):
    # checksum, counter, and quality flag checks
    for quality_flag in [True, False]:
      for msg_type in ["speed", "speed_2", "yaw"]:
        self.safety.set_controls_allowed(True)
        # send multiple times to verify counter checks
        for _ in range(10):
          if msg_type == "speed":
            msg = self._speed_msg(0, quality_flag=quality_flag)
          elif msg_type == "speed_2":
            msg = self._speed_msg_2(0, quality_flag=quality_flag)
          elif msg_type == "yaw":
            msg = self._yaw_rate_msg(0, 0, quality_flag=quality_flag)

          self.assertEqual(quality_flag, self._rx(msg))
          self.assertEqual(quality_flag, self.safety.get_controls_allowed())

        # Mess with checksum to make it fail, checksum is not checked for 2nd speed
        msg[0].data[3] = 0  # Speed checksum & half of yaw signal
        should_rx = msg_type == "speed_2" and quality_flag
        self.assertEqual(should_rx, self._rx(msg))
        self.assertEqual(should_rx, self.safety.get_controls_allowed())

  def test_angle_measurements(self):
    """Tests rx hook correctly parses the curvature measurement from the vehicle speed and yaw rate"""
    for speed in np.arange(0.5, 40, 0.5):
      for curvature in np.arange(0, self.MAX_CURVATURE * 2, 2e-3):
        self._rx(self._speed_msg(speed))
        for c in (curvature, -curvature, 0, 0, 0, 0):
          self._rx(self._yaw_rate_msg(c, speed))

        self.assertEqual(self.safety.get_curvature_meas_min(), round(-curvature * self.DEG_TO_CAN))
        self.assertEqual(self.safety.get_curvature_meas_max(), round(curvature * self.DEG_TO_CAN))

        self._rx(self._yaw_rate_msg(0, speed))
        self.assertEqual(self.safety.get_curvature_meas_min(), round(-curvature * self.DEG_TO_CAN))
        self.assertEqual(self.safety.get_curvature_meas_max(), 0)

        self._rx(self._yaw_rate_msg(0, speed))
        self.assertEqual(self.safety.get_curvature_meas_min(), 0)
        self.assertEqual(self.safety.get_curvature_meas_max(), 0)

  def test_max_lateral_acceleration(self):
    # Ford CAN FD can achieve a higher max lateral acceleration than CAN so we limit curvature based
    # on speed. On CAN the only bound is the curvature signal range -- see limit_lateral_accel.
    for speed in np.arange(0, 40, 0.5):
      max_can = self._max_curvature_allowed_can(speed)
      for offset in (-5, -1, 0, 1, 5):
        curvature_can = max_can + offset
        curvature = curvature_can / self.DEG_TO_CAN

        for sign in (-1, 1):
          signed_curvature = sign * curvature
          self.safety.set_controls_allowed(True)
          self._set_prev_desired_angle(signed_curvature)
          self._reset_curvature_measurement(signed_curvature, speed)

          should_tx = abs(curvature_can) <= max_can
          self.assertEqual(should_tx, self._tx(self._lat_ctl_msg(True, 0, 0, signed_curvature, 0)))

  def test_steer_allowed(self):
    path_offsets = np.arange(-5.12, 5.11, 2.5).round()
    path_angles = np.arange(-0.5, 0.5235, 0.25).round(1)
    curvature_rates = np.arange(-0.001024, 0.00102375, 0.001).round(3)
    curvatures = np.arange(-0.02, 0.02094, 0.01).round(2)

    for speed in (self.CURVATURE_ERROR_MIN_SPEED - 1,
                  self.CURVATURE_ERROR_MIN_SPEED + 1):
      max_curvature = self._max_curvature_allowed_can(speed) / self.DEG_TO_CAN
      for controls_allowed in (True, False):
        for steer_control_enabled in (True, False):
          for path_offset in path_offsets:
            for path_angle in path_angles:
              for curvature_rate in curvature_rates:
                for curvature in curvatures:
                  def msg(enabled=steer_control_enabled, po=path_offset, pa=path_angle,
                          c=curvature, cr=curvature_rate):
                    return self._lat_ctl_msg(enabled, po, pa, c, cr)

                  self._reset_curvature_measurement(curvature, speed)
                  # BluePilot: path_offset, path_angle and curvature_rate each carry their own
                  # rate-of-change limit, so a value that steps from the previous sweep iteration
                  # would be blocked on the transition frame alone. Send the frame once to settle
                  # those limiters, then assert on an identical frame -- this test is about the
                  # value ranges, the rate limits are covered by test_curvature_rate_limits.
                  self.safety.set_controls_allowed(controls_allowed)
                  self._tx(msg())
                  self.safety.set_controls_allowed(controls_allowed)
                  self._set_prev_desired_angle(curvature)

                  if steer_control_enabled:
                    # BluePilot: unlike upstream, openpilot drives path_offset, path_angle and
                    # curvature_rate as real signals -- they are bounded, not pinned to zero.
                    # Limits mirror FORD_PATH_OFFSET/PATH_ANGLE/CURVATURE_RATE_MIN/MAX in ford.h.
                    # angle_mode_engaged is not set here, so path_angle keeps its tight cap.
                    should_tx = (controls_allowed and
                                 -1.0 <= path_offset <= 1.0 and
                                 -0.25 <= path_angle <= 0.25 and
                                 -0.001024 <= curvature_rate <= 0.00102375 and
                                 abs(curvature) <= max_curvature)
                  else:
                    # when the request bit is 0 every lateral signal must sit at its neutral value;
                    # the signal ranges are not large enough to enforce them tracking measured
                    should_tx = (path_offset == 0 and path_angle == 0 and
                                 curvature_rate == 0 and curvature == 0)

                  with self.subTest(controls_allowed=controls_allowed, steer_control_enabled=steer_control_enabled,
                                    path_offset=float(path_offset), path_angle=float(path_angle), curvature_rate=float(curvature_rate),
                                    curvature=float(curvature)):
                    self.assertEqual(should_tx, self._tx(msg()))

  def test_curvature_rate_limits(self):
    """
    When the curvature error is exceeded, commanded curvature must start moving towards meas respecting rate limits.
    Since safety allows higher rate limits to avoid false positives, we need to allow a lower rate to move towards meas.
    """
    self.safety.set_controls_allowed(True)
    # safety fudges the speed (1 m/s) and rate limits (1 CAN unit) to avoid false positives
    small_curvature = 1 / self.DEG_TO_CAN  # significant small amount of curvature to cross boundary
    # step this far past each boundary before asserting, see RATE_LIMIT_TOL_CAN
    tol = self.RATE_LIMIT_TOL_CAN / self.DEG_TO_CAN

    for speed in np.arange(0, 40, 0.5):
      curvature_accel_limit = self._max_curvature_allowed_can(speed) / self.DEG_TO_CAN
      limit_command = speed > self.CURVATURE_ERROR_MIN_SPEED
      # ensure our limits match the safety's rounded limits
      # the wind up and wind down tables are identical in ford.h, so the limits are symmetric
      max_delta = self._get_max_curvature_delta_can(speed) / self.DEG_TO_CAN
      max_delta_relaxed = self._get_max_curvature_delta_relaxed_can(speed) / self.DEG_TO_CAN

      # the error band edge sits at (measured - MAX_CURVATURE_ERROR), so it moves with any
      # uncertainty in the measurement -- step clear of it in both directions
      meas_tol = self._meas_tol_can(speed) * small_curvature
      band_inside = self.MAX_CURVATURE_ERROR - small_curvature + meas_tol
      band_outside = self.MAX_CURVATURE_ERROR - small_curvature * 2 - meas_tol

      up_cases = (self.MAX_CURVATURE_ERROR * 2, [
        (not limit_command, 0, 0),
        (not limit_command, 0, max_delta_relaxed - tol - small_curvature),
        (True, 0, max_delta_relaxed + tol),
        (True, 0, max_delta - tol),
        (False, 0, max_delta + tol + small_curvature),
        # stay at boundary limit
        (True, band_inside, band_inside),
        # below boundary limit
        (not limit_command, band_outside, band_outside),
        # shouldn't allow command to move outside the boundary limit if last was inside
        (not limit_command, band_inside, band_outside),
      ])

      down_cases = (self.MAX_CURVATURE - self.MAX_CURVATURE_ERROR * 2, [
        (not limit_command, self.MAX_CURVATURE, self.MAX_CURVATURE),
        (not limit_command, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_relaxed + tol + small_curvature),
        (True, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_relaxed - tol),
        (True, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta + tol),
        (False, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta - tol - small_curvature),
      ])

      # the driver can hold a curvature openpilot may not command, safety must never require moving
      # past the most it can send -- on this path that is the curvature signal range itself
      max_curvature_relaxed = self._get_max_curvature_relaxed_can(speed) / self.DEG_TO_CAN
      max_curvature_allowed_can = round(self.MAX_CURVATURE * self.DEG_TO_CAN)
      relaxed_cases = (self.MAX_CURVATURE * 2, [
        (True, max_curvature_relaxed, max_curvature_relaxed),
        (not limit_command, max_curvature_relaxed, max_curvature_relaxed - small_curvature),
        # no longer requiring the command to wind towards meas doesn't stop rate limiting it winding away
        (True, max_curvature_allowed_can / self.DEG_TO_CAN, max_curvature_relaxed),
      ])

      for sign in (-1, 1):
        for angle_meas, cases in (up_cases, down_cases, relaxed_cases):
          self._reset_curvature_measurement(sign * angle_meas, speed)
          for should_tx, initial_curvature, desired_curvature in cases:

            # at low speeds one frame of jerk exceeds the curvature signal, so the should_tx=False cases will rightly not fail.
            # assert we never drop a case at a speed where the curvature error is enforced
            if abs(desired_curvature) > self.MAX_CURVATURE:
              self.assertLess(speed, self.CURVATURE_ERROR_MIN_SPEED)
              continue

            # can not send if the curvature is above the max lateral acceleration
            should_tx = should_tx and abs(desired_curvature) <= curvature_accel_limit

            self._set_prev_desired_angle(sign * initial_curvature)
            self.assertEqual(should_tx, self._tx(self._lat_ctl_msg(True, 0, 0, sign * desired_curvature, 0)))

    # the newest speed sample gates the check, not the whole sample window
    max_error = self.MAX_CURVATURE_ERROR + small_curvature * 2
    for sign in (-1, 1):
      self._reset_curvature_measurement(0, self.CURVATURE_ERROR_MIN_SPEED - 1)
      self._rx(self._speed_msg(self.CURVATURE_ERROR_MIN_SPEED + 1))
      self._set_prev_desired_angle(sign * self.MAX_CURVATURE_ERROR)
      self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, sign * max_error, 0)))

  def test_curvature_violation(self):
    # If violation occurs, curvature cmd is blocked until reset to 0
    self.safety.set_controls_allowed(True)
    speed = 25.
    max_delta_can = self._get_max_curvature_delta_can(speed)
    self._reset_curvature_measurement(0, speed)

    self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0))
    over_curvature = (max_delta_can + 5) / self.DEG_TO_CAN
    for _ in range(20):
      self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, over_curvature, 0)))
    self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0)))

    # prev tracks the commanded curvature on a passing tx (not reset to 0 every frame)
    self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, max_delta_can / self.DEG_TO_CAN, 0)))
    self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, 2 * max_delta_can / self.DEG_TO_CAN, 0)))

  def test_rt_limits(self):
    # send rate is limited over a rolling 250ms window split into two half-interval buckets
    self.safety.set_controls_allowed(True)
    self._reset_curvature_measurement(0, 0)
    max_rt_msgs = int(self.LATERAL_FREQUENCY * common.RT_INTERVAL / 1e6 * 1.2 + 1)
    half = common.RT_INTERVAL // 2

    # too many messages within one window is blocked
    self.safety.set_timer(0)
    for i in range(max_rt_msgs * 2):
      self.assertEqual(i <= max_rt_msgs, self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))

    # shift the overflow into the previous bucket
    self.safety.set_timer(half)
    self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))

    # previous bucket still counts within the half interval
    self.safety.set_timer(half + 2 * common.RT_INTERVAL // 5)
    self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))
    self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))

    # both buckets clear after a full interval
    self.safety.set_timer(half + common.RT_INTERVAL)
    self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))
    self.safety.set_timer(half + 2 * common.RT_INTERVAL)
    for _ in range(max_rt_msgs):
      self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, 0, 0, increment_timer=False)))

  def test_angle_mode_corroboration_gate(self):
    """The wide angle-mode path_angle range (FORD_DBC_PATH_ANGLE_MIN/MAX, the full DBC range) must
    be unlocked only by Lane_Assist_Data1's angle_mode_engaged bit. The angle-mode sentinel on its
    own -- desired_curvature == 0 -- must not be enough: in curvature mode path_angle keeps the
    tight FORD_PATH_ANGLE_MIN/MAX cap, so a frame cannot claim the wide range just by zeroing
    curvature.

    Note this asserts the corroboration that ford.h actually implements. It deliberately does NOT
    assert that every curvature == 0 frame is blocked when angle_mode_engaged is clear: 0 is the
    inactive curvature sentinel, so that would forbid openpilot from commanding straight ahead in
    ordinary curvature mode."""
    self.safety.set_controls_allowed(True)
    # outside FORD_PATH_ANGLE_MAX (0.25), inside FORD_DBC_PATH_ANGLE_MAX (0.5235)
    wide_path_angle = 0.4
    for speed in (5.0, 15.0):
      for angle_mode_engaged in (True, False):
        self._reset_curvature_measurement(0, speed)
        self._tx(self._lka_bp_status_msg(angle_mode_engaged, 0.0))
        # settle path_angle's own rate limiter; only the value range is under test here
        self._tx(self._lat_ctl_msg(True, 0, wide_path_angle, 0, 0))
        self.safety.set_controls_allowed(True)
        with self.subTest(speed=speed, angle_mode_engaged=angle_mode_engaged):
          self.assertEqual(angle_mode_engaged, self._tx(self._lat_ctl_msg(True, 0, wide_path_angle, 0, 0)))

  def test_angle_mode_sentinel_keeps_tight_path_angle_cap(self):
    """Sanity companion to test_angle_mode_corroboration_gate: with angle mode NOT corroborated, a
    curvature == 0 frame is still perfectly legal inside the tight path_angle cap."""
    self.safety.set_controls_allowed(True)
    for speed in (5.0, 15.0):
      self._reset_curvature_measurement(0, speed)
      self._tx(self._lka_bp_status_msg(False, 0.0))
      self._tx(self._lat_ctl_msg(True, 0, 0.01, 0, 0))
      self.safety.set_controls_allowed(True)
      with self.subTest(speed=speed):
        self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0.01, 0, 0)))

  def test_curvature_mode_unaffected_by_angle_mode_flag(self):
    """Real (nonzero) curvature commands are self-evidently curvature mode by their own content --
    they must never be gated by Lane_Assist_Data1's angle_mode_engaged, regardless of its value."""
    self.safety.set_controls_allowed(True)
    speed = 15.0
    curvature = 0.01
    self._reset_curvature_measurement(curvature, speed)
    for angle_mode_engaged in (True, False):
      self._set_prev_desired_angle(curvature)
      self._tx(self._lka_bp_status_msg(angle_mode_engaged, 0.0))
      with self.subTest(angle_mode_engaged=angle_mode_engaged):
        self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, curvature, 0)))

  def test_shadow_curvature_deviation_check(self):
    """Angle mode's shadow_curvature must be checked against measured curvature (angle_meas),
    gated the same way as curvature mode's own check: enforce_angle_error + CURVATURE_ERROR_MIN_SPEED.
    Mirrors test_curvature_rate_limits' up/down structure but for the deviation-only path.
    path_angle held at a small nonzero value -- see test_angle_mode_corroboration_gate docstring."""
    self.safety.set_controls_allowed(True)
    for speed in (self.CURVATURE_ERROR_MIN_SPEED - 1, self.CURVATURE_ERROR_MIN_SPEED + 1):
      limit_enforced = speed > self.CURVATURE_ERROR_MIN_SPEED
      measured_curvature = 0.005
      self._reset_curvature_measurement(measured_curvature, speed)

      self._tx(self._lka_bp_status_msg(True, measured_curvature))
      with self.subTest(speed=speed, case="matches_measured"):
        self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0.01, 0, 0)))

      large_deviation = measured_curvature + (self.MAX_CURVATURE_ERROR * 5)
      self._tx(self._lka_bp_status_msg(True, large_deviation))
      with self.subTest(speed=speed, case="large_deviation"):
        self.assertEqual(not limit_enforced, self._tx(self._lat_ctl_msg(True, 0, 0.01, 0, 0)))

  def test_shadow_curvature_no_rate_limit(self):
    """shadow_curvature must NOT be rate-of-change limited -- only path_angle's own ROC
    (path_angle_cmd_checks) applies in angle mode. A large frame-to-frame jump in shadow_curvature,
    while it stays within deviation tolerance of a correspondingly-updated measured curvature, must
    not block. Regression test for a real bug: substituting shadow_curvature into
    the full curvature check (which does both ROC and deviation) caused spurious blocks from
    shadow_curvature's own frame-to-frame movement, unrelated to path_angle's actual behavior.
    path_angle held at a small nonzero value -- see test_angle_mode_corroboration_gate docstring."""
    self.safety.set_controls_allowed(True)
    speed = self.CURVATURE_ERROR_MIN_SPEED + 1
    for curvature in (0.015, -0.015, 0.018, -0.018, 0.001, -0.019):
      self._reset_curvature_measurement(curvature, speed)
      self._tx(self._lka_bp_status_msg(True, curvature))
      with self.subTest(curvature=curvature):
        self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0.01, 0, 0)))

  def test_prevent_lkas_action(self):
    self.safety.set_controls_allowed(1)
    self.assertFalse(self._tx(self._lkas_command_msg(1)))

    self.safety.set_controls_allowed(0)
    self.assertFalse(self._tx(self._lkas_command_msg(1)))

  def test_acc_buttons(self):
    for allowed in (0, 1):
      self.safety.set_controls_allowed(allowed)
      for enabled in (True, False):
        self._rx(self._pcm_status_msg(enabled))
        self.assertTrue(self._tx(self._acc_button_msg(Buttons.TJA_TOGGLE, 2)))

    for allowed in (0, 1):
      self.safety.set_controls_allowed(allowed)
      for bus in (0, 2):
        self.assertEqual(allowed, self._tx(self._acc_button_msg(Buttons.RESUME, bus)))

    for enabled in (True, False):
      self._rx(self._pcm_status_msg(enabled))
      for bus in (0, 2):
        self.assertEqual(enabled, self._tx(self._acc_button_msg(Buttons.CANCEL, bus)))

  def test_enable_control_allowed_from_acc_main_on(self):
    for enable_mads in (True, False):
      with self.subTest("enable_mads", mads_enabled=enable_mads):
        for main_button_msg_valid in (True, False):
          with self.subTest("main_button_msg_valid", state_valid=main_button_msg_valid):
            self.safety.set_mads_params(enable_mads, False, False)
            self._rx(self._pcm_status_msg(main_button_msg_valid))
            self.assertEqual(enable_mads and main_button_msg_valid, self.safety.get_controls_allowed_lateral())


class TestFordCANFDStockSafety(TestFordSafetyBase):
  STEER_MESSAGE = MSG_LateralMotionControl2

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl2, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.CANFD)
    self.safety.init_tests()


class TestFordLongitudinalSafetyBase(TestFordSafetyBase):
  MAX_ACCEL = 2.0  # accel is used for brakes, but openpilot can set positive values
  MIN_ACCEL = -3.5
  INACTIVE_ACCEL = 0.0

  MAX_GAS = 2.0
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  # ACC command
  def _acc_command_msg(self, gas: float, brake: float, brake_actuation: bool, cmbb_deny: bool = False):
    values = {
      "AccPrpl_A_Rq": gas,                              # [-5|5.23] m/s^2
      "AccPrpl_A_Pred": gas,                            # [-5|5.23] m/s^2
      "AccBrkTot_A_Rq": brake,                          # [-20|11.9449] m/s^2
      "AccBrkPrchg_B_Rq": 1 if brake_actuation else 0,  # Pre-charge brake request: 0=No, 1=Yes
      "AccBrkDecel_B_Rq": 1 if brake_actuation else 0,  # Deceleration request: 0=Inactive, 1=Active
      "CmbbDeny_B_Actl": 1 if cmbb_deny else 0,         # [0|1] deny AEB actuation
    }
    return self.packer.make_can_msg_safety("ACCDATA", 0, values)

  def test_stock_aeb(self):
    # Test that CmbbDeny_B_Actl is never 1, it prevents the ABS module from actuating AEB requests from ACCDATA_2
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for cmbb_deny in (True, False):
        should_tx = not cmbb_deny
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.INACTIVE_GAS, self.INACTIVE_ACCEL, controls_allowed, cmbb_deny)))
        should_tx = controls_allowed and not cmbb_deny
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.MAX_GAS, self.MAX_ACCEL, controls_allowed, cmbb_deny)))

  def test_gas_safety_check(self):
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for gas in np.concatenate((np.arange(self.MIN_GAS - 2, self.MAX_GAS + 2, 0.05), [self.INACTIVE_GAS])):
        gas = round(gas, 2)  # floats might not hit exact boundary conditions without rounding
        should_tx = (controls_allowed and self.MIN_GAS <= gas <= self.MAX_GAS) or gas == self.INACTIVE_GAS
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(gas, self.INACTIVE_ACCEL, controls_allowed)))

  def test_brake_safety_check(self):
    brake_values = self._boundary_values([self.MIN_ACCEL, self.MAX_ACCEL, self.INACTIVE_ACCEL],
                                         self.MIN_ACCEL - 2, self.MAX_ACCEL + 2, 0.05)
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for brake_actuation in (True, False):
        for brake in brake_values:
          should_tx = (controls_allowed and self.MIN_ACCEL <= brake <= self.MAX_ACCEL) or brake == self.INACTIVE_ACCEL
          should_tx = should_tx and (controls_allowed or not brake_actuation)
          self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.INACTIVE_GAS, brake, brake_actuation)))


class TestFordLongitudinalSafety(TestFordLongitudinalSafetyBase):
  STEER_MESSAGE = MSG_LateralMotionControl

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA, 0], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    # Make sure we enforce long safety even without long flag for CAN
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, 0)
    self.safety.init_tests()


class TestFordCANFDLongitudinalSafety(TestFordLongitudinalSafetyBase):
  STEER_MESSAGE = MSG_LateralMotionControl2

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA, 0], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl2, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.LONG_CONTROL | FordSafetyFlags.CANFD)
    self.safety.init_tests()


# =============================================================================
# BluePilot: steering-angle curvature measurement (FordSafetyFlagsSP.STEER_ANGLE_CURVATURE)
#
# Opt-in alternative angle_meas source for vehicles whose RCM broadcasts implausible yaw
# while its quality flag reads OK. The classes below run the ENTIRE stock test matrix with
# angle_meas sourced from SteeringPinion_Data and the widened 0.003 error band, plus
# pinion-specific tests. The stock (flag-off) classes above never set SAFETY_PARAM_SP, so
# their outcomes (including any pre-existing failures) must stay bit-identical to the base
# branch -- that comparison is the default-off zero-delta check.
# =============================================================================

class TestFordPinionCurvatureSafetyBase(TestFordSafetyBase):
  MAX_CURVATURE_ERROR = 0.003  # widened: raw pinion angle has no roll/offset compensation in firmware

  # Per-platform geometry (see the *PinionGeometry mixins). Values must match the
  # ford_pinion_geometry row for GEOMETRY_INDEX -- the table itself is checked against
  # CarSpecs + calc_slip_factor by TestFordPinionGeometryTable, so these literals only
  # need to agree with that already-verified table.
  GEOMETRY_INDEX = 0
  PINION_SLIP_FACTOR = 0.0
  PINION_STEER_RATIO = 1.0
  PINION_WHEELBASE = 1.0

  cnt_pinion = 0

  def _curvature_to_pinion_angle_deg(self, curvature: float, speed: float) -> float:
    # Inverse of the firmware conversion in ford_rx_hook (modes/ford.h):
    # curvature = angle_rad * curvature_factor(speed) / steer_ratio
    speed = max(speed, 0.1)
    curvature_factor = 1. / (1. - (self.PINION_SLIP_FACTOR * (speed ** 2))) / self.PINION_WHEELBASE
    angle_rad = curvature * self.PINION_STEER_RATIO / curvature_factor
    return float(np.degrees(angle_rad))

  def _meas_tol_can(self, speed):
    return self._pinion_quant_tol(speed)

  def _pinion_quant_tol(self, speed: float) -> int:
    # 0.1 deg DBC quantization -> curvature CAN units at this speed (+2 for float rounding)
    speed = max(speed, 0.1)
    curvature_factor = 1. / (1. - (self.PINION_SLIP_FACTOR * (speed ** 2))) / self.PINION_WHEELBASE
    return int(np.radians(0.1) * curvature_factor / self.PINION_STEER_RATIO * self.DEG_TO_CAN) + 2

  # Current curvature measurement (pinion-angle sourced, not yaw)
  def _pinion_msg(self, curvature: float, speed: float, quality_flag=True):
    values = {"StePinComp_An_Est": self._curvature_to_pinion_angle_deg(curvature, speed),
              "StePinCompAnEst_D_Qf": 3 if quality_flag else 0,
              "StePinAn_No_Cnt": self.cnt_pinion % 16}
    self.__class__.cnt_pinion += 1
    return self.packer.make_can_msg_safety("SteeringPinion_Data", 0, values)

  def _reset_curvature_measurement(self, curvature, speed):
    # 14 frames, not 6: frames after a counter discontinuity (e.g. rejected bad-QF frames
    # advanced the python-side counter) are dropped by the rx counter check until it
    # re-syncs, which would otherwise leave stale samples in the 6-deep angle_meas buffer
    for _ in range(14):
      self._rx(self._speed_msg(speed))
      # the second speed source must be kept in sync too -- steer_curvature_cmd_checks runs
      # speed_mismatch_check on every lateral tx, and a stale vehicle_speed_2 drops
      # controls_allowed, which would block every command this helper is setting up for
      self._rx(self._speed_msg_2(speed))
      self._rx(self._pinion_msg(curvature, speed))

  def test_rx_hook(self):
    # checksum, counter, and quality flag checks (stock matrix + the pinion message)
    for quality_flag in [True, False]:
      for msg_type in ["speed", "speed_2", "yaw", "pinion"]:
        self.safety.set_controls_allowed(True)
        # send multiple times to verify counter checks
        for _ in range(10):
          if msg_type == "speed":
            msg = self._speed_msg(0, quality_flag=quality_flag)
          elif msg_type == "speed_2":
            msg = self._speed_msg_2(0, quality_flag=quality_flag)
          elif msg_type == "yaw":
            msg = self._yaw_rate_msg(0, 0, quality_flag=quality_flag)
          elif msg_type == "pinion":
            msg = self._pinion_msg(0, 0, quality_flag=quality_flag)

          self.assertEqual(quality_flag, self._rx(msg))
          self.assertEqual(quality_flag, self.safety.get_controls_allowed())

        # Mess with checksum to make it fail; checksum is not checked for 2nd speed or pinion
        # (pinion has an unknown OEM checksum algorithm; integrity is via counter + quality flag)
        msg[0].data[3] = 0  # Speed checksum & half of yaw/pinion angle signal
        should_rx = msg_type in ("speed_2", "pinion") and quality_flag
        self.assertEqual(should_rx, self._rx(msg))
        self.assertEqual(should_rx, self.safety.get_controls_allowed())

  def test_angle_measurements(self):
    """Tests rx hook correctly parses the curvature measurement from the steering pinion angle.

    The DBC signal quantizes to 0.1 deg, so allow the quantization-equivalent CAN-unit
    tolerance from the round trip through the packer.
    """
    for speed in np.arange(0.5, 40, 0.5):
      for curvature in np.arange(0, self.MAX_CURVATURE * 2, 2e-3):
        self._rx(self._speed_msg(speed))
        for c in (curvature, -curvature, 0, 0, 0, 0):
          self._rx(self._pinion_msg(c, speed))

        quant_tol = self._pinion_quant_tol(speed)
        self.assertAlmostEqual(self.safety.get_curvature_meas_min(), round(-curvature * self.DEG_TO_CAN), delta=quant_tol)
        self.assertAlmostEqual(self.safety.get_curvature_meas_max(), round(curvature * self.DEG_TO_CAN), delta=quant_tol)

        self._rx(self._pinion_msg(0, speed))
        self.assertAlmostEqual(self.safety.get_curvature_meas_min(), round(-curvature * self.DEG_TO_CAN), delta=quant_tol)
        self.assertAlmostEqual(self.safety.get_curvature_meas_max(), 0, delta=quant_tol)

        self._rx(self._pinion_msg(0, speed))
        self.assertAlmostEqual(self.safety.get_curvature_meas_min(), 0, delta=quant_tol)
        self.assertAlmostEqual(self.safety.get_curvature_meas_max(), 0, delta=quant_tol)

  def test_pinion_quality_flag_gates_measurement(self):
    """A bad pinion quality flag must reject the message (measurement not updated)."""
    speed = self.CURVATURE_ERROR_MIN_SPEED + 5
    self._reset_curvature_measurement(0.005, speed)
    meas_max_before = self.safety.get_curvature_meas_max()
    self.assertGreater(meas_max_before, 0)

    # bad-QF frames must be rejected at rx and leave angle_meas untouched
    for _ in range(6):
      self.assertFalse(self._rx(self._pinion_msg(0, speed, quality_flag=False)))
    self.assertEqual(self.safety.get_curvature_meas_max(), meas_max_before)

  def test_pinion_sign_convention(self):
    """Command matching the measured curvature sign passes the error check; a sign-inverted
    command (the broken-yaw failure mode) violates above the gate speed."""
    speed = self.CURVATURE_ERROR_MIN_SPEED + 5
    curvature = 0.005  # well above MAX_CURVATURE_ERROR so the inverted case must violate

    for sign in (1, -1):
      with self.subTest(sign=sign):
        self._reset_curvature_measurement(sign * curvature, speed)
        self.safety.set_controls_allowed(True)
        self._set_prev_desired_angle(sign * curvature)
        # matching-sign command: allowed
        self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, sign * curvature, 0)))
        # inverted command (what a sign-flipped sensor would demand): blocked
        self._set_prev_desired_angle(-sign * curvature)
        self.assertFalse(self._tx(self._lat_ctl_msg(True, 0, 0, -sign * curvature, 0)))

  def test_pinion_check_inert_below_gate_speed(self):
    """Below CURVATURE_ERROR_MIN_SPEED the deviation check must not constrain commands."""
    self.safety.set_controls_allowed(True)
    speed = self.CURVATURE_ERROR_MIN_SPEED - 2
    self._reset_curvature_measurement(0.005, speed)
    # command far from measured, but below gate: allowed (rate limits still apply, so seed prev)
    inverted = -0.005
    self._set_prev_desired_angle(inverted)
    self.assertTrue(self._tx(self._lat_ctl_msg(True, 0, 0, inverted, 0)))


class FordExplorerPinionGeometry:
  """FORD_EXPLORER_MK6 -- the on-road-validated primary platform."""
  GEOMETRY_INDEX = 5
  PINION_SLIP_FACTOR = -0.00055447339
  PINION_STEER_RATIO = 16.8
  PINION_WHEELBASE = 3.025
  SAFETY_PARAM_SP = int(FordSafetyFlagsSP.STEER_ANGLE_CURVATURE) | (GEOMETRY_INDEX << FORD_PINION_GEOMETRY_SHIFT)


class FordBroncoSportPinionGeometry:
  """FORD_BRONCO_SPORT_MK1 -- smallest wheelbase in the table."""
  GEOMETRY_INDEX = 1
  PINION_SLIP_FACTOR = -0.00062819555
  PINION_STEER_RATIO = 17.7
  PINION_WHEELBASE = 2.670
  SAFETY_PARAM_SP = int(FordSafetyFlagsSP.STEER_ANGLE_CURVATURE) | (GEOMETRY_INDEX << FORD_PINION_GEOMETRY_SHIFT)


class FordF150PinionGeometry:
  """FORD_F_150_MK14 -- largest wheelbase in the table."""
  GEOMETRY_INDEX = 8
  PINION_SLIP_FACTOR = -0.00042037149
  PINION_STEER_RATIO = 17.0
  PINION_WHEELBASE = 3.990
  SAFETY_PARAM_SP = int(FordSafetyFlagsSP.STEER_ANGLE_CURVATURE) | (GEOMETRY_INDEX << FORD_PINION_GEOMETRY_SHIFT)


class TestFordPinionLongitudinalSafety(FordExplorerPinionGeometry, TestFordPinionCurvatureSafetyBase, TestFordLongitudinalSafety):
  pass


class TestFordPinionCANFDStockSafety(FordExplorerPinionGeometry, TestFordPinionCurvatureSafetyBase, TestFordCANFDStockSafety):
  pass


class TestFordPinionCANFDLongitudinalSafety(FordExplorerPinionGeometry, TestFordPinionCurvatureSafetyBase, TestFordCANFDLongitudinalSafety):
  pass


class TestFordPinionBroncoSportSafety(FordBroncoSportPinionGeometry, TestFordPinionCurvatureSafetyBase, TestFordLongitudinalSafety):
  pass


class TestFordPinionF150Safety(FordF150PinionGeometry, TestFordPinionCurvatureSafetyBase, TestFordCANFDLongitudinalSafety):
  pass


class TestFordPinionGeometryTable(unittest.TestCase):
  """The firmware geometry table must match CarSpecs + calc_slip_factor(VehicleModel(CP))
  for every supported platform, so the table cannot rot as platforms change. Reads the
  table through the ALLOW_DEBUG libsafety getters -- no header parsing."""

  TX_MSGS: list = []  # not a CarSafetyTest; keeps common.py's cross-mode TX sweep happy

  def test_geometry_matches_carspecs(self):
    safety = libsafety_py.libsafety
    count = safety.get_ford_pinion_geometry_count()
    self.assertEqual(count, len(FORD_PINION_GEOMETRY_INDEX))
    # the index rides bits 1-4 of current_safety_param_sp; growing past 15 would silently
    # disable the firmware side while the control side still enables -- never allow it
    self.assertLessEqual(count, 15)

    seen = set()
    for car in CAR:
      if car.config.flags & FordFlags.ALT_STEER_ANGLE:
        # relative pinion angle with a learned offset -- unsupported by design
        self.assertNotIn(car, FORD_PINION_GEOMETRY_INDEX)
        continue
      self.assertIn(car, FORD_PINION_GEOMETRY_INDEX, f"{car} has no geometry-table row")
      idx = FORD_PINION_GEOMETRY_INDEX[car]
      self.assertTrue(1 <= idx <= count, f"{car}: index {idx} out of range")
      self.assertNotIn(idx, seen, f"{car}: duplicate index {idx}")
      seen.add(idx)

      specs = car.config.specs
      CP = CarParams()
      CP.mass = specs.mass
      CP.wheelbase = specs.wheelbase
      CP.steerRatio = specs.steerRatio
      CP.centerToFront = specs.wheelbase * specs.centerToFrontRatio
      CP.tireStiffnessFactor = specs.tireStiffnessFactor
      CP.tireStiffnessFront, CP.tireStiffnessRear = scale_tire_stiffness(
        CP.mass, CP.wheelbase, CP.centerToFront, CP.tireStiffnessFactor)
      slip_factor = calc_slip_factor(VehicleModel(CP))

      self.assertAlmostEqual(safety.get_ford_pinion_geometry_steer_ratio(idx), specs.steerRatio, places=3, msg=str(car))
      self.assertAlmostEqual(safety.get_ford_pinion_geometry_wheelbase(idx), specs.wheelbase, places=3, msg=str(car))
      self.assertAlmostEqual(safety.get_ford_pinion_geometry_slip_factor(idx), slip_factor,
                             delta=abs(slip_factor) * 1e-4, msg=str(car))

  def test_invalid_index_row_is_inert(self):
    # index 0 is the reserved invalid row: zero slip, unit ratios
    safety = libsafety_py.libsafety
    self.assertEqual(safety.get_ford_pinion_geometry_slip_factor(0), 0.0)
    self.assertEqual(safety.get_ford_pinion_geometry_steer_ratio(0), 1.0)
    self.assertEqual(safety.get_ford_pinion_geometry_wheelbase(0), 1.0)


if __name__ == "__main__":
  unittest.main()
