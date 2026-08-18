"""BluePilot: regression tests for the Ford radar interfaces.

These exist because of a real field crash. BluePilot's STEER_ASSIST_DATA branch assigned
`aRel`, `yvRel` and `measured` directly on a RadarPoint. Upstream had since moved all three
into RadarPoint's `deprecated` group in car.capnp, so the assignment raised

    AttributeError: capnp/schema.c++:511: failed: struct has no such member; name = aRel

which killed `card` mid-drive. Nothing caught it: the branch only runs on a CAN FD Ford once
the forward sensor reports a lead with CmbbObjConfdnc_D_Stat > 0, so every parked test and the
whole stationary calibration session sailed past it.

The schema-shape tests below are deliberately written against the schema rather than against a
hardcoded field list, so that if upstream deprecates another RadarPoint field the tests fail
here instead of on the road.
"""

import unittest

from opendbc.car import structs
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.radar_interface import RadarInterface
from opendbc.car.ford.values import CAR, RADAR

# Fields a RadarPoint is allowed to have assigned at the top level. Anything a radar_interface
# writes that is not in here has been deprecated upstream and will raise at runtime.
LIVE_RADAR_POINT_FIELDS = {'trackId', 'dRel', 'yRel', 'vRel'}


def _params(platform):
  CP = CarInterface.get_non_essential_params(str(platform))
  try:
    CP_SP = CarInterface.get_non_essential_params_sp(CP, str(platform))
  except AttributeError:
    CP_SP = structs.CarParamsSP()
  return CP, CP_SP


class TestRadarPointSchema(unittest.TestCase):
  def test_deprecated_fields_are_not_top_level(self):
    """The exact shape that broke us: these three must NOT be settable top-level."""
    pt = structs.RadarData.RadarPoint()
    for name in ('aRel', 'yvRel', 'measured'):
      with self.subTest(field=name):
        assert not hasattr(pt, name), \
          f"RadarPoint.{name} is top-level again -- revisit ford/radar_interface.py"

  def test_live_field_set_matches_expectation(self):
    """If upstream deprecates another field, fail here rather than mid-drive."""
    pt = structs.RadarData.RadarPoint()
    live = {f for f in pt.schema.fieldnames if f != 'deprecated'}
    assert live == LIVE_RADAR_POINT_FIELDS, \
      f"RadarPoint top-level fields changed: {live} != {LIVE_RADAR_POINT_FIELDS}. Audit every assignment in ford/radar_interface.py."


class TestSteerAssistData(unittest.TestCase):
  """Drive the branch that crashed, on the platform that hit it."""

  def setUp(self):
    CP, CP_SP = _params(CAR.FORD_MUSTANG_MACH_E_MK1)
    self.RI = RadarInterface(CP, CP_SP)
    assert self.RI.radar == RADAR.STEER_ASSIST_DATA, "Mach-E must route to STEER_ASSIST_DATA"

  def _feed(self, dRel, confidence, yRel=0.0, vRel=0.0, yvRel=0.0):
    self.RI.rcp.vl["Steer_Assist_Data"] = {
      'CmbbObjDistLong_L_Actl': dRel,
      'CmbbObjConfdnc_D_Stat': confidence,
      'CmbbObjDistLat_L_Actl': yRel,
      'CmbbObjRelLong_V_Actl': vRel,
      'CmbbObjRelLat_V_Actl': yvRel,
    }
    return self.RI._update_steer_assist_data()

  def test_lead_detected_does_not_raise(self):
    """The regression. Pre-fix this raised AttributeError on aRel."""
    assert self._feed(dRel=30.0, confidence=3) is True
    assert 0 in self.RI.pts
    assert self.RI.pts[0].dRel == 30.0

  def test_point_populates_live_fields(self):
    self._feed(dRel=25.0, confidence=3, yRel=1.5, vRel=-2.0)
    self._feed(dRel=25.0, confidence=3, yRel=1.5, vRel=-2.0)  # second frame: not a new track
    pt = self.RI.pts[0]
    assert pt.dRel == 25.0
    assert pt.yRel == 1.5
    assert pt.vRel == -2.0

  def test_no_confidence_drops_the_track(self):
    self._feed(dRel=30.0, confidence=3)
    assert 0 in self.RI.pts
    self._feed(dRel=30.0, confidence=0)
    assert 0 not in self.RI.pts

  def test_repeated_updates_are_stable(self):
    """Exercise the new-track / same-track / track-reassignment paths together."""
    for i in range(50):
      self._feed(dRel=40.0 - i * 0.5, confidence=3, vRel=-1.0)
    assert 0 in self.RI.pts
    for _ in range(10):
      self._feed(dRel=80.0, confidence=3, vRel=5.0)  # big jump -> new trackId
    assert 0 in self.RI.pts


class TestAllFordRadarsConstruct(unittest.TestCase):
  def test_every_platform_builds_a_radar_interface(self):
    for platform in CAR:
      with self.subTest(platform=str(platform)):
        CP, CP_SP = _params(platform)
        RadarInterface(CP, CP_SP)


if __name__ == '__main__':
  unittest.main()
