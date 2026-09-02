"""Shared deterministic checks for creator-facing spoken-script style."""

from __future__ import annotations

import re


_CLAUSE_START = r"(?:^|[\r\n。！？；!?;])\s*"
_CHINESE_ORDINAL = r"(?:[一二三四五六七八九十]+|\d+)"
_ABSTRACT_SECTION = r"(?:因果|逻辑|原因|问题|框架|分析|利弊|权衡|机制|取舍)"

OUTLINE_LABEL_PATTERN = re.compile(
    _CLAUSE_START
    + r"(?:"
    + r"(?:结尾记忆点|开场钩子|开头钩子|核心观点|中心观点|权衡框架|"
    + r"论证框架|内容总结|总结金句|金句总结|适用边界|行动建议|本段结论|"
    + r"最终结论)\s*[:：]"
    + r"|"
    + rf"(?:(?:先|再|接着|然后)\s*)?(?:(?:拆|分析|进入|展开)\s*)?"
    + rf"第{_CHINESE_ORDINAL}层(?:\s*{_ABSTRACT_SECTION}|"
    + rf"\s*(?:看|讲|说|给).{{0,24}}{_ABSTRACT_SECTION})"
    + r")",
    re.IGNORECASE | re.MULTILINE,
)


__all__ = ["OUTLINE_LABEL_PATTERN"]
