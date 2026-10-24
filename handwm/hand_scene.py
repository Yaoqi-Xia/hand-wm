"""Builds the five-finger scene: free-floating Shadow Hand (MuJoCo Menagerie) + up to 8 blocks.

The Menagerie file is read, not modified. We drop the forearm, hang the wrist under a new
`hand_root` body with x/y/z slides, a yaw hinge and a roll hinge about the hand's long axis.
At roll 0 the hand is a vertical "karate chop": palm vertical, fingers pointing forward,
little finger down, thumb up - finger curls happen in the horizontal plane and are clearly
visible from the top camera. Roll -pi/2 = palm down, +pi/2 = palm up. Height (z) is set by
the env so the hand never enters the table.

Contact bits: table (1,2), walls (1,0), blocks (1,3), hand (2,0)
  -> hand touches blocks and the table (not walls, not itself); blocks touch everything.
"""
from __future__ import annotations

import colorsys
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHADOW_DIR = ROOT / "third_party" / "menagerie" / "shadow_hand"

MAX_BLOCKS = 8
BLOCK_HALF = 0.025
ARENA_HALF = 0.24          # walls at +-ARENA_HALF
PALM_HEIGHT = 0.05         # height of the palm centre above the table
VIEW_EXTENT = 0.52         # top camera sees VIEW_EXTENT x VIEW_EXTENT metres
# 8 fully saturated hues 45 degrees apart: block colour == block identity in images.
# (Hand and table are grey/black, so hue alone separates blocks from everything else.)
BLOCK_HUES = np.arange(MAX_BLOCKS) * 45.0
BLOCK_COLORS = np.array([colorsys.hsv_to_rgb(h / 360.0, 1.0, 0.85) for h in BLOCK_HUES])

# palm frame -> world at yaw = roll = 0 (vertical chop, little finger down, thumb up):
# fingers (+z_palm) -> +x, index side (+x_palm) -> +z, palm front (-y_palm) -> +y
_R_PALM = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
_PALM_CENTRE_IN_WRIST = np.array([0.0, 0.0, 0.034 + 0.045])


# distal bodies and the distance from their origin to the fingertip
TIP_BODIES = {"rh_thdistal": 0.032, "rh_ffdistal": 0.026, "rh_mfdistal": 0.026,
              "rh_rfdistal": 0.026, "rh_lfdistal": 0.026}
# MediaPipe hand-landmark order -> (kind, name) on the Shadow Hand
KEYPOINTS = (
    [("body", "rh_palm")]
    + [("body", "rh_thbase"), ("body", "rh_thhub"), ("body", "rh_thdistal"), ("site", "rh_thdistal_tip")]
    + [x for f in ("ff", "mf", "rf", "lf")
       for x in (("body", f"rh_{f}proximal"), ("body", f"rh_{f}middle"),
                 ("body", f"rh_{f}distal"), ("site", f"rh_{f}distal_tip"))]
)
ROLL_LIMIT = 1.75          # rad, about 100 degrees each way
FINGER_STIFFNESS = 4.0     # multiplier on Shadow's finger servo gain and force limit
CONTACT_SOLREF = "0.004 1"
CONTACT_SOLIMP = "0.95 0.99 0.001"


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def build_hand_xml(n_block_slots: int = MAX_BLOCKS, ghosts: bool = False) -> str:
    """ghosts=True adds 8 translucent, non-colliding mocap boxes "ghost0..7" (display only;
    they add no joints, so qpos has the same layout as the simulation model)."""
    src = ET.parse(SHADOW_DIR / "right_hand.xml").getroot()

    root = ET.Element("mujoco", model="hand_blocks")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(SHADOW_DIR / "assets"),
                  autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", cone="elliptic", impratio="10")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="720")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.4 0.4 0.4",
                  specular="0 0 0")

    # defaults: copy Shadow's, then make hand collision geoms touch blocks only
    default = copy.deepcopy(src.find("default"))
    for d in default.iter("default"):
        if d.get("class") == "plastic_collision":
            d.find("geom").set("contype", "2")
            d.find("geom").set("conaffinity", "0")
    # Stiffer contacts everywhere (MuJoCo's defaults let pushed blocks sink ~1-2 cm into each
    # other): time constant 4 ms (= 2 x timestep), impedance close to 1 from the start.
    ET.SubElement(default, "geom", solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    for d in default.iter("default"):
        if d.get("class") == "plastic":
            d.find("geom").set("solref", CONTACT_SOLREF)
            d.find("geom").set("solimp", CONTACT_SOLIMP)
    # Shadow's finger servos are very soft (kp ~1, max 1 Nm): a block could bend a finger
    # sideways and through the table. Make them FINGER_STIFFNESS x stiffer and stronger.
    for p in default.iter("position"):
        p.set("kp", _fmt([float(p.get("kp", 1.0)) * FINGER_STIFFNESS]))
        if p.get("forcerange"):
            p.set("forcerange", _fmt(np.array(p.get("forcerange").split(), float) * FINGER_STIFFNESS))
    root.append(default)

    asset = copy.deepcopy(src.find("asset"))
    for m in list(asset.findall("mesh")):
        if m.get("file", "").startswith("forearm"):
            asset.remove(m)
    ET.SubElement(asset, "material", name="table", rgba="0.42 0.42 0.42 1")
    ET.SubElement(asset, "material", name="wall", rgba="0.30 0.30 0.30 1")
    root.append(asset)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", directional="true",
                  castshadow="false", diffuse="0.5 0.5 0.5")
    # table: conaffinity 2 -> the hand can touch it (backstop; the env keeps the hand above it)
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 0.01",
                  material="table", contype="1", conaffinity="2")
    h = 0.03
    for i, (px, py, sx, sy) in enumerate([
        (ARENA_HALF + 0.01, 0, 0.01, ARENA_HALF + 0.02), (-ARENA_HALF - 0.01, 0, 0.01, ARENA_HALF + 0.02),
        (0, ARENA_HALF + 0.01, ARENA_HALF + 0.02, 0.01), (0, -ARENA_HALF - 0.01, ARENA_HALF + 0.02, 0.01),
    ]):
        ET.SubElement(world, "geom", name=f"wall{i}", type="box", pos=_fmt([px, py, h]),
                      size=_fmt([sx, sy, h]), material="wall", contype="1", conaffinity="0")

    for i in range(n_block_slots):
        b = ET.SubElement(world, "body", name=f"block{i}", pos=_fmt([2 + 0.2 * i, 2, BLOCK_HALF]))
        ET.SubElement(b, "freejoint", name=f"block{i}")
        ET.SubElement(b, "geom", name=f"block{i}", type="box", size=_fmt([BLOCK_HALF] * 3),
                      mass="0.1", rgba=_fmt([*BLOCK_COLORS[i], 1]), friction="0.5 0.01 0.001",
                      contype="1", conaffinity="3")

    if ghosts:
        for i in range(n_block_slots):
            g = ET.SubElement(world, "body", name=f"ghost{i}", mocap="true", pos=_fmt([2 + 0.2 * i, 3, 0]))
            ET.SubElement(g, "geom", name=f"ghost{i}", type="box", size=_fmt([BLOCK_HALF * 1.02] * 3),
                          rgba=_fmt([*BLOCK_COLORS[i], 0.28]), contype="0", conaffinity="0")

    # hand root: its origin is the palm centre. Joints (applied in this order):
    #   x, y, z slides; yaw about vertical; roll about the hand's long axis
    #   (roll 0 = vertical chop, -pi/2 = palm down, +pi/2 = palm up).
    # z is not part of the action: the env lifts the hand so it never enters the table.
    hr = ET.SubElement(world, "body", name="hand_root", pos="0 0 0", gravcomp="1")
    ET.SubElement(hr, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.002 0.002 0.002")
    ET.SubElement(hr, "joint", name="hand_x", type="slide", axis="1 0 0", damping="40")
    ET.SubElement(hr, "joint", name="hand_y", type="slide", axis="0 1 0", damping="40")
    ET.SubElement(hr, "joint", name="hand_z", type="slide", axis="0 0 1", damping="150")
    ET.SubElement(hr, "joint", name="hand_yaw", type="hinge", axis="0 0 1", damping="1",
                  armature="0.01")
    ET.SubElement(hr, "joint", name="hand_roll", type="hinge", axis="1 0 0", damping="1",
                  armature="0.01")
    ET.SubElement(hr, "site", name="palm_centre", size="0.005", rgba="1 0 0 0")

    forearm = src.find("worldbody/body")              # rh_forearm
    wrist = copy.deepcopy(forearm.find("body"))       # rh_wrist and everything below
    wrist.set("quat", _fmt(_mat2quat(_R_PALM)))
    wrist.set("pos", _fmt(-_R_PALM @ _PALM_CENTRE_IN_WRIST))
    wrist.set("childclass", "right_hand")
    for body in wrist.iter("body"):
        body.set("gravcomp", "1")
        if body.get("name") in TIP_BODIES:  # fingertip sites for the 21 keypoints
            ET.SubElement(body, "site", name=body.get("name") + "_tip",
                          pos=_fmt([0, 0, TIP_BODIES[body.get("name")]]), size="0.003",
                          group="4")
    hr.append(wrist)

    for name, pos, fov, extra in [
        ("top", [0, 0, 1.0], VIEW_EXTENT, {"projection": "orthographic", "xyaxes": "1 0 0 0 1 0"}),
        ("persp", [0, -0.55, 0.55], 45, {"mode": "targetbody", "target": "view_centre"}),
    ]:
        ET.SubElement(world, "camera", name=name, pos=_fmt(pos), fovy=str(fov), **extra)
    ET.SubElement(world, "body", name="view_centre", pos="0 0 0")

    if ghosts:  # display only: translucent copy of the hand ("g_" + every name), visual geoms only
        gh = copy.deepcopy(hr)
        for el in gh.iter():
            if el.get("name"):
                el.set("name", "g_" + el.get("name"))
        for parent in list(gh.iter()):
            for ch in list(parent):
                if ch.tag == "site" or (ch.tag == "geom" and ch.get("class") != "plastic_visual"):
                    parent.remove(ch)          # no collision geoms, no sites
                elif ch.tag == "geom":
                    ch.attrib.pop("material", None)
                    ch.set("rgba", "0.85 0.85 0.95 0.3")
                    ch.set("contype", "0")
                    ch.set("conaffinity", "0")
        world.append(gh)                       # last in the tree: its joints come after all others

    contact = ET.SubElement(root, "contact")
    for ex in src.find("contact"):
        if "rh_forearm" not in (ex.get("body1"), ex.get("body2")):
            contact.append(copy.deepcopy(ex))
    root.append(copy.deepcopy(src.find("tendon")))

    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "position", name="A_hand_x", joint="hand_x", kp="600",
                  forcerange="-20 20")
    ET.SubElement(act, "position", name="A_hand_y", joint="hand_y", kp="600",
                  forcerange="-20 20")
    # stiff upward (lifts the hand faster than the fingers can curl down) but only 15 N
    # downward: a palm-down hand coming onto a block rests on it instead of crushing it
    ET.SubElement(act, "position", name="A_hand_z", joint="hand_z", kp="5000",
                  forcerange="-15 300")
    ET.SubElement(act, "position", name="A_hand_yaw", joint="hand_yaw", kp="20",
                  forcerange="-5 5")
    ET.SubElement(act, "position", name="A_hand_roll", joint="hand_roll", kp="20",
                  forcerange="-5 5")
    shadow_act = copy.deepcopy(src.find("actuator"))
    for a in shadow_act:
        a.set("class", a.get("class"))  # keep Shadow's classes (kp, ranges)
        act.append(a)

    return ET.tostring(root, encoding="unicode")


# ---------------- finger coupling: one bend value per finger ----------------
FINGERS = ["thumb", "index", "middle", "ring", "little"]


def bend_targets(bends) -> dict[str, float]:
    """bends: 5 values in [0, 1] (0 = straight, 1 = fully curled) -> actuator targets."""
    th, ff, mf, rf, lf = np.clip(np.asarray(bends, dtype=float), 0.0, 1.0)
    t = {
        "rh_A_WRJ2": 0.0, "rh_A_WRJ1": 0.0,
        # thumb: swing across the palm and curl
        "rh_A_THJ5": 0.6 * th, "rh_A_THJ4": 1.1 * th, "rh_A_THJ3": 0.0,
        "rh_A_THJ2": 0.4 * th, "rh_A_THJ1": 1.2 * th,
        "rh_A_LFJ5": 0.3 * lf,
    }
    for prefix, b in (("FF", ff), ("MF", mf), ("RF", rf), ("LF", lf)):
        t[f"rh_A_{prefix}J4"] = 0.0
        t[f"rh_A_{prefix}J3"] = 1.5 * b      # knuckle
        t[f"rh_A_{prefix}J0"] = 2.6 * b      # middle + distal (tendon-coupled)
    return t


# Canonical hand shapes for the scripted policy: [thumb, index, middle, ring, little]
HAND_SHAPES = {
    "open":  [0.0, 0.0, 0.0, 0.0, 0.0],
    "fist":  [0.9, 1.0, 1.0, 1.0, 1.0],
    "poke":  [0.9, 0.0, 1.0, 1.0, 1.0],
    "pinch": [0.75, 0.55, 1.0, 1.0, 1.0],
}
SHAPE_NAMES = list(HAND_SHAPES)
