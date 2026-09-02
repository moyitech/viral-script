"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { clamp, magnetize, pointerValue } = require("../../app/desktop/assets/slider.js");

const RANGE = {
  width: 900,
  minimum: 100,
  maximum: 1000,
  snapPoints: [280, 450, 700],
  enterPixels: 14,
  releasePixels: 24,
};

function magnetic(rawValue, attachedPoint = null) {
  return magnetize({ ...RANGE, rawValue, attachedPoint });
}

test("clamp and pointerValue keep pointer input within the range", () => {
  assert.equal(clamp(-1, 0, 10), 0);
  assert.equal(clamp(11, 0, 10), 10);
  assert.equal(pointerValue(20, 100, 900, 100, 1000), 100);
  assert.equal(pointerValue(550, 100, 900, 100, 1000), 550);
  assert.equal(pointerValue(1200, 100, 900, 100, 1000), 1000);
  assert.equal(pointerValue(200, 100, 0, 100, 1000), 100);
});

test("all three magnetic points capture pointer movement at the entry edge", () => {
  for (const point of RANGE.snapPoints) {
    assert.deepEqual(magnetic(point - 14), { value: point, attachedPoint: point });
    assert.deepEqual(magnetic(point), { value: point, attachedPoint: point });
    assert.deepEqual(magnetic(point + 14), { value: point, attachedPoint: point });
  }
});

test("movement outside the entry threshold remains continuous", () => {
  assert.deepEqual(magnetic(295), { value: 295, attachedPoint: null });
  assert.deepEqual(magnetic(425), { value: 425, attachedPoint: null });
  assert.deepEqual(magnetic(725), { value: 725, attachedPoint: null });
});

test("an attached point stays magnetic until the wider release edge is crossed", () => {
  assert.deepEqual(magnetic(474, 450), { value: 450, attachedPoint: 450 });
  assert.deepEqual(magnetic(475, 450), { value: 475, attachedPoint: null });
  assert.deepEqual(magnetic(426, 450), { value: 450, attachedPoint: 450 });
  assert.deepEqual(magnetic(425, 450), { value: 425, attachedPoint: null });
});
