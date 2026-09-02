"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const css = fs.readFileSync(path.join(root, "app/desktop/assets/styles.css"), "utf8");
const javascript = fs.readFileSync(path.join(root, "app/desktop/assets/app.js"), "utf8");

test("the document viewport is fixed and cannot become the scroll container", () => {
  assert.match(css, /html, body\s*\{[^}]*height:\s*100%[^}]*overflow:\s*hidden/s);
  assert.match(css, /body\s*\{[^}]*position:\s*fixed[^}]*inset:\s*0/s);
  assert.match(css, /main\s*\{[^}]*min-height:\s*0[^}]*overflow:\s*hidden/s);
  assert.doesNotMatch(javascript, /window\.scrollTo\s*\(/);
});

test("long stage content scrolls only inside its dedicated region", () => {
  for (const selector of [
    ".recommendation-list",
    ".log-console",
    ".script-body",
    "#report-view",
  ]) {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    assert.match(css, new RegExp(`${escaped}\\s*\\{[^}]*overflow-y:\\s*auto`, "s"));
  }
});

test("the initial compose card never becomes an internal scroll container", () => {
  assert.match(css, /\.composer-panel\s*\{[^}]*overflow:\s*hidden/s);
  assert.doesNotMatch(css, /\.composer-panel\s*\{[^}]*overflow-y:\s*auto/s);
});
