#!/usr/bin/env python3
"""Check RoboJudo joystick axis adaptation for UniLab policy control."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


DEFAULT_ROBOJUDO_ROOT = Path("/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real")


class FakeJoystick:
    def __init__(self, num_axes: int):
        self._num_axes = num_axes

    def get_numaxes(self) -> int:
        return self._num_axes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    args = parser.parse_args()

    joystick_path = args.robojudo_root.resolve() / "robojudo/controller/utils/joystick.py"
    spec = importlib.util.spec_from_file_location("robojudo_joystick_axis_test", joystick_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {joystick_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    JoystickThread = module.JoystickThread

    axis_map = {
        "LeftX": 0,
        "LeftY": 1,
        "RightX": 3,
        "RightY": 4,
        "LT": 2,
        "RT": 5,
    }
    axis_range = {"LT": [0, 1], "RT": [0, 1]}
    invert = {"LeftY", "RightY"}

    four_map, four_range, four_invert = JoystickThread._adapt_axis_map(
        axis_map, axis_range, invert, FakeJoystick(4)
    )
    six_map, six_range, six_invert = JoystickThread._adapt_axis_map(
        axis_map, axis_range, invert, FakeJoystick(6)
    )

    print("four_axis_map:", four_map)
    print("four_axis_range:", four_range)
    print("four_axis_invert:", sorted(four_invert))
    print("six_axis_map:", six_map)
    print("six_axis_range:", six_range)
    print("six_axis_invert:", sorted(six_invert))

    fail = 0
    if four_map != {"LeftX": 0, "LeftY": 1, "RightX": 2, "RightY": 3}:
        fail += 1
        print("[FAIL] four_axis_mapping")
    else:
        print("[PASS] four_axis_mapping")
    if four_range != {}:
        fail += 1
        print("[FAIL] four_axis_drops_triggers")
    else:
        print("[PASS] four_axis_drops_triggers")
    if four_invert != {"LeftY", "RightY"}:
        fail += 1
        print("[FAIL] four_axis_invert")
    else:
        print("[PASS] four_axis_invert")
    if six_map != axis_map or six_range != axis_range or six_invert != invert:
        fail += 1
        print("[FAIL] six_axis_mapping_preserved")
    else:
        print("[PASS] six_axis_mapping_preserved")

    print(f"summary: {fail} fail(s), {4 - fail} pass(es)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
