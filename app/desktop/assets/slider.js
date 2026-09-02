(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.HySlider = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function pointerValue(clientX, left, width, minimum, maximum) {
    if (!Number.isFinite(width) || width <= 0) return minimum;
    const ratio = clamp((clientX - left) / width, 0, 1);
    return Math.round(minimum + ratio * (maximum - minimum));
  }

  function magnetize(options) {
    const { rawValue, width, minimum, maximum, snapPoints, attachedPoint,
      enterPixels, releasePixels } = options;
    const pixelsPerUnit = width / (maximum - minimum);
    const pixelDistance = (point) => Math.abs(rawValue - point) * pixelsPerUnit;
    if (attachedPoint !== null && snapPoints.includes(attachedPoint)) {
      if (pixelDistance(attachedPoint) <= releasePixels) {
        return { value: attachedPoint, attachedPoint };
      }
    }
    let nearest = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const point of snapPoints) {
      const distance = pixelDistance(point);
      if (distance < nearestDistance) { nearest = point; nearestDistance = distance; }
    }
    if (nearest !== null && nearestDistance <= enterPixels) {
      return { value: nearest, attachedPoint: nearest };
    }
    return { value: rawValue, attachedPoint: null };
  }

  return { clamp, pointerValue, magnetize };
});
