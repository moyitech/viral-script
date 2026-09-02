"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const topicDefaults = require("../../app/desktop/assets/topic_defaults.js");

test("offline defaults contain multiple distinct usable topics", () => {
  assert.ok(topicDefaults.topics.length >= 10);
  assert.equal(new Set(topicDefaults.topics).size, topicDefaults.topics.length);
  for (const topic of topicDefaults.topics) {
    assert.equal(typeof topic, "string");
    assert.ok(topic.trim().length > 0);
    assert.ok(topic.length <= 200);
  }
});

test("choose can select the full offline pool deterministically", () => {
  assert.equal(topicDefaults.choose(() => 0), topicDefaults.topics[0]);
  assert.equal(
    topicDefaults.choose(() => 0.999999),
    topicDefaults.topics[topicDefaults.topics.length - 1],
  );
});

test("resolve prefers typed content and otherwise uses the displayed example", () => {
  assert.equal(topicDefaults.resolve("  用户自己的选题  ", "默认选题"), "用户自己的选题");
  assert.equal(topicDefaults.resolve("   ", "  默认选题  "), "默认选题");
});
