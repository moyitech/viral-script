(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.HyTopicDefaults = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const topics = Object.freeze([
    "年轻人为什么开始重新喜欢逛菜市场？",
    "AI 工具普及后，普通人的核心竞争力会变成什么？",
    "预制菜是效率升级，还是吃饭体验的退步？",
    "远程办公让人更自由，还是更难真正下班？",
    "情绪价值为什么正在影响越来越多消费选择？",
    "年轻人开始认真存钱，是更理性了还是更缺安全感？",
    "低价竞争最终会让消费者受益，还是让选择变少？",
    "当算法比你更懂喜好，我们的选择是变多还是变少？",
    "短视频时代，我们为什么越来越难耐心看完一件事？",
    "旅游打卡越来越卷，我们究竟是在记录生活还是完成任务？",
    "城市里的公共空间，为什么会影响普通人的幸福感？",
    "自动驾驶普及之前，事故责任应该怎样划分？",
  ]);

  function choose(random = Math.random) {
    const sampled = Number(random());
    const unit = Number.isFinite(sampled) ? Math.min(0.999999999, Math.max(0, sampled)) : 0;
    return topics[Math.floor(unit * topics.length)];
  }

  function resolve(input, fallback) {
    const normalizedInput = typeof input === "string" ? input.trim() : "";
    if (normalizedInput) return normalizedInput;
    return typeof fallback === "string" ? fallback.trim() : "";
  }

  return { topics, choose, resolve };
});
