"""Plan, execute, and revise live-search queries while recording lineage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
import re
from typing import Any, Literal, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from hyscript.config import ResearchConfig
from hyscript.llm import (
    AsyncLLMClient,
    ChatMessage,
    LLMCallUsage,
    LLMProviderError,
    llm_call_usage,
)
from hyscript.llm.prompts import (
    BACKGROUND_SELECTION_VERSION,
    RESEARCH_EVIDENCE_PROMPT_VERSION,
    RESEARCH_EVIDENCE_SYSTEM_PROMPT,
    RESEARCH_QUERY_PLAN_PROMPT_VERSION,
    RESEARCH_QUERY_PLAN_SYSTEM_PROMPT,
)
from hyscript.search import (
    AsyncSearchProvider,
    SearchProviderError,
    SearchResponse,
    SearchResult,
)

from ._structured import (
    StructuredOutputError,
    json_object,
    required_text,
    text_list,
)
from .contracts import (
    CORE_SOURCE_TYPES_BY_CLAIM_KIND,
    Claim,
    ClaimKind,
    ClaimSupportStatus,
    Evidence,
    EvidenceSourceType,
    PlannedQuery,
    QueryPlan,
    ResearchOutcome,
    ScriptTask,
    TitleChainPart,
)

_AssessmentStatus = Literal["ready", "needs_more", "insufficient_evidence"]
logger = logging.getLogger(__name__)
_RESEARCH_TIMEZONE = ZoneInfo("Asia/Shanghai")
_MAX_BLOCK_LENGTH = 400
_BLOCK_BOUNDARY_PATTERN = re.compile(
    r"(?:[。！？；]+[”’」』】）)\]]*|"
    r"[.!?;]+[\"')\]]*(?=\s|$)|\r?\n+)"
)
_ATTRIBUTION_SOURCE_PATTERN = r"[^《》“”\"'，。；！？,:：\r\n]{2,60}?"
_ATTRIBUTION_VERB_PATTERN = (
    r"(?:文章(?:指出|显示|认为|称)|记者梳理发现|回应(?:称|表示|指出)|"
    r"调查(?:显示|发现|表明)|"
    r"报道显示|报告显示|通报显示|报道称|数据表明|数据显示|"
    r"研究表明|研究显示|研究发现|表示|指出|披露|发布|认为|宣称|声称|称)"
)
_BASED_ON_REPORT_ATTRIBUTION_PATTERN = re.compile(
    r"(?:^|[，。；！？,:：\r\n])(?:据|根据)?\s*[《“\"]?"
    rf"(?P<source>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*基于\s*[《“\"]?"
    rf"(?P<underlying>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*(?:的\s*)?报告\s*"
    rf"{_ATTRIBUTION_VERB_PATTERN}"
)
_COMPOSITE_ATTRIBUTION_PATTERN = re.compile(
    r"(?:^|[，。；！？,:：\r\n])(?:据|根据)?\s*[《“\"]?"
    rf"(?P<outlet>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*(?:的\s*)?(?:文章|报道)\s*"
    r"(?:援引|引述|引用|引)\s*[《“\"]?"
    rf"(?P<speaker>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*"
    rf"{_ATTRIBUTION_VERB_PATTERN}"
)
_PUBLISHED_MATERIAL_ATTRIBUTION_PATTERN = re.compile(
    r"(?:^|[，。；！？,:：\r\n])(?:据|根据)\s*[《“\"]?"
    rf"(?P<source>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*(?:发布|公布|披露|印发)\s*的\s*"
    r"(?:通报|公告|报告|数据|资料|材料|说明|研究)"
)
_LEADING_SOURCE_VERB_PATTERN = re.compile(
    r"(?:^|[，。；！？,:：\r\n])(?:据|根据)\s*[《“\"]?"
    rf"(?P<source>{_ATTRIBUTION_SOURCE_PATTERN})"
    r"[》”\"]?\s*(?:报道称|表示|指出|披露|认为|声称|称)"
)
_ATTRIBUTION_PATTERNS = (
    re.compile(
        r"(?:^|[，。；！？,:：\r\n])(?:据|根据)\s*[《“\"]?"
        rf"(?P<source>{_ATTRIBUTION_SOURCE_PATTERN})"
        r"[》”\"]?\s*(?:的\s*)?"
        r"(?:报道|消息|通报|数据|资料|材料|公告|说明|披露|统计|研究)"
    ),
    re.compile(
        r"(?:^|[，。；！？,:：\r\n])(?!据|根据)[《“\"]?"
        rf"(?P<source>{_ATTRIBUTION_SOURCE_PATTERN})"
        r"[》”\"]?\s*"
        rf"{_ATTRIBUTION_VERB_PATTERN}"
    ),
)
_GENERIC_ATTRIBUTION_SOURCES = frozenset(
    {
        "媒体",
        "某媒体",
        "有媒体",
        "有评论",
        "某评论",
        "相关评论",
        "相关媒体",
        "多家媒体",
        "媒体分析",
        "有媒体分析",
        "某媒体分析",
        "相关媒体分析",
        "公开材料",
        "公开报道",
        "有关报道",
        "相关报道",
        "研究简报",
        "独立二手材料",
        "公开",
        "相关",
        "有关",
        "现有",
        "官方",
        "专家",
        "教育专家",
        "研究人员",
        "业内人士",
        "有人",
        "一篇",
        "一项",
        "一份",
        "论文摘要",
    }
)
_GENERIC_ATTRIBUTION_PATTERNS = (
    re.compile(
        r"^(?:(?:一|这|该)(?:篇|项|份))?"
        r"(?:媒体|系统综述|综述|研究|论文|报告|文章)$"
    ),
    re.compile(
        r"^(?:(?:某|相关|该)(?:家)?)?(?:厂商|企业)"
        r"(?:宣传材料|宣传|材料)?$"
    ),
    re.compile(
        r"^(?:社区|平台|学校|行业)(?:管理)?(?:规范|规则|规定)"
        r"(?:明确)?(?:禁止|要求|规定)$"
    ),
    re.compile(
        r"^(?:19|20)\d{2}年一项对\d+名[^，。；！？,:：\r\n]{1,40}"
        r"(?:追踪|研究|调查)(?:\([^)]+\))?$"
    ),
    re.compile(
        r"^基于[^，。；！？,:：\r\n]{0,30}\d+(?:名|万多笔|万笔|笔|份|个)"
        r"[^，。；！？,:：\r\n]{0,30}(?:订单|样本|记录|案例)"
        r"(?:\([^)]+\))?$"
    ),
    re.compile(
        r"^全国多地(?:中小学|中学|学校)(?:在[^，。；！？,:：\r\n]{1,20})?$"
    ),
    re.compile(
        r"^(?:拍摄|上传|公开|发布|传播)[^，。；！？,:：\r\n]{1,50}"
        r"(?:属于|是|为)[^，。；！？,:：\r\n]{1,30}$"
    ),
    re.compile(
        r"^(?:一名|一位|一家)(?:互联网)?(?:医疗|购物|外卖)?平台(?:的)?用户"
        r"[^，。；！？,:：\r\n]{1,40}$"
    ),
    re.compile(r"^(?:医学|学术|相关)?研究$"),
)
_ACADEMIC_TEAM_SOURCE_PATTERN = re.compile(
    r"^(?P<institution>.+?(?:大学|学院|研究院|研究所|实验室))(?:的)?"
    r"(?:学者|教授|研究人员)"
    r"(?P<people>[\u4e00-\u9fff]{2,4}(?:[、，][\u4e00-\u9fff]{2,4})*)$"
)
_CONTEXTUAL_PERSON_SOURCE_PATTERN = re.compile(
    r"^(?:一名|一位|来自).*(?P<person>[\u4e00-\u9fff]{1,2}(?:女士|先生|教授|博士|医生))$"
)
_COURT_DECISION_SOURCE_PATTERN = re.compile(
    r"^(?:\d{4}年)?(?P<court>[^，。；！？,:：\r\n]{2,30}?法院)"
    r"(?:(?:一审|二审|再审)?(?:判决|裁定))$"
)
_PERSON_ACTION_SOURCE_PATTERN = re.compile(
    r"^(?P<person>[\u4e00-\u9fff]{2,4})在某[^，。；！？,:：\r\n]+$"
)
_PERSON_DATED_COMMENTARY_SOURCE_PATTERN = re.compile(
    r"^(?P<person>[\u4e00-\u9fff]{2,4})(?:19|20)\d{2}年"
    r"(?:\d{1,2}月(?:\d{1,2}日)?)?(?:分析|观点)$"
)
_LATIN_BLOG_SOURCE_PATTERN = re.compile(
    r"^(?P<organization>[A-Z][A-Za-z0-9]*(?:[ ._-][A-Za-z0-9]+)*)博客$"
)
_LATIN_DOCUMENT_SOURCE_PATTERN = re.compile(
    r"^(?P<organization>[A-Z][A-Za-z0-9]*(?:[ ._-][A-Za-z0-9]+)*)"
    r"(?:报告|数据|研究|综述|公告|材料)$"
)
_NAMED_VENDOR_ROLE_SOURCE_PATTERN = re.compile(
    r"^(?:厂商|企业|平台)(?P<organization>[\u4e00-\u9fffA-Za-z0-9 ._-]{2,30})$"
)
_CHINESE_DOCUMENT_SOURCE_PATTERN = re.compile(
    r"^(?P<organization>[\u4e00-\u9fff]{2,30}?)(?:规则|报告|公告|材料|数据)$"
)
_LATIN_ROLE_SUFFIX_SOURCE_PATTERN = re.compile(
    r"^(?P<organization>[A-Z][A-Za-z0-9]*(?:[ ._-][A-Za-z0-9]+)*)"
    r"(?:平台|网站|系统|数据库)$"
)
_NAMED_RULE_ACTION_SOURCE_PATTERN = re.compile(
    r"^(?P<rule>.+?(?:公约|规则|规定|规范|办法|条例|制度))"
    r"(?:明确)?(?:禁止|要求|规定)$"
)
_LATIN_REPORT_SOURCE_PATTERN = re.compile(
    r"^(?P<organization>[A-Z][A-Za-z0-9]*(?:[ ._-][A-Za-z0-9]+)*)\s+"
    r"\d{4}(?:年)?[^，。；！？,:：\r\n]{0,40}$"
)
_ATTRIBUTION_FREQUENCY_SUFFIX_PATTERN = re.compile(
    r"^(?P<source>.+?)(?:曾经?|多次|反复|近期|近日|此前|日前)$"
)
_MEDIA_BASED_STUDY_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>[\u4e00-\u9fffA-Za-z0-9 ._-]{2,30}?"
    r"(?:评论|日报|时报|商报|新闻|周刊|杂志|电视台|通讯社|广播|网))"
    r"基于[^，。；！？,:：\r\n]{1,80}"
    r"(?:样本|数据|订单|记录|案例)(?:研究)?$"
)
_OUTLET_ANONYMOUS_SPEAKER_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>.+?)(?:引述|援引|引用)(?:一名|一位)?"
    r"(?:不愿具名|匿名)(?:的)?[^，。；！？,:：\r\n]{1,20}$"
)
_NAMED_SCHOOL_GROUP_SOURCE_PATTERN = re.compile(
    r"^(?P<school>.+?(?:大学|学院|中学|小学|学校))"
    r"等多地(?:中小学|中学|小学|学校)$"
)
_NAMED_SURFACE_MATERIAL_SOURCE_PATTERN = re.compile(
    r"^(?P<source>.+?(?:平台|网站|系统|数据库))(?:记录|数据|资料|案例)$"
)
_DIRECT_COMPOSITE_NAMED_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>[^，。；！？,:：\r\n]{2,30}?)"
    r"(?:引述|援引|引用|引)"
    r"(?P<speaker>[^，。；！？,:：\r\n]{2,30})$"
)
_RESEARCH_ORGANIZATION_ACTION_SOURCE_PATTERN = re.compile(
    r"^(?P<institution>.+?(?:研究中心|研究院|研究所|实验室|大学|学院))"
    r"(?:分析|研究|调查)(?:我国|中国|全国)?\d+"
    r"(?:个|座|所|名|份|组)[^，。；！？,:：\r\n]{0,30}$"
)
_INSTITUTION_PERSON_SOURCE_PATTERN = re.compile(
    r"^(?P<institution>.+?(?:医院|大学|学院|研究院|研究所|研究中心|实验室))"
    r"(?P<person>[\u4e00-\u9fff]{2,4})$"
)
_MEDIA_RELAY_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>.+?(?:评论|日报|时报|商报|新闻|周刊|杂志|电视台|通讯社|广播|网|报|社))"
    r"(?:文章)?(?:转述|援引|引述|引用|引)"
    r"(?P<underlying>[^，。；！？,:：\r\n]{2,30})$"
)
_MEDIA_REPORTS_DOCUMENT_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>.+?(?:评论|日报|时报|商报|新闻|周刊|杂志|电视台|通讯社|广播|网|报|社))"
    r"报道(?P<underlying>[^，。；！？,:：\r\n]{2,30}(?:公约|规则|规定|规范|办法|条例))$"
)
_DATED_SURVEY_ATTRIBUTION_SOURCE_PATTERN = re.compile(
    r"^(?P<outlet>.+?)(?:19|20)\d{2}年(?:调查|研究)\d+"
    r"(?:份|名|个|组)[^，。；！？,:：\r\n]{0,40}$"
)

_NAMED_SOURCE_SUFFIXES = (
    "委员会",
    "研究中心",
    "研究院",
    "研究所",
    "实验室",
    "通讯社",
    "电视台",
    "工作局",
    "办公室",
    "出版社",
    "大学",
    "学院",
    "医院",
    "法院",
    "学校",
    "中学",
    "小学",
    "公司",
    "集团",
    "平台",
    "网站",
    "数据库",
    "杂志",
    "周刊",
    "日报",
    "时报",
    "商报",
    "新闻",
    "评论",
    "公约",
    "条例",
    "办法",
    "规范",
    "规定",
    "总台",
    "电台",
    "广播",
    "报",
    "网",
    "社",
    "局",
    "部",
)


def _log_text(value: str) -> str:
    """Collapse untrusted text before placing it on one console log line."""

    return " ".join(value.split())[:200]


def _normalized_source_identity(value: str) -> str:
    """Normalize visible source labels without guessing aliases."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _is_generic_attribution_source(value: str) -> bool:
    """Return whether an attribution label is a tightly bounded generic phrase."""

    normalized = value.strip()
    return normalized in _GENERIC_ATTRIBUTION_SOURCES or any(
        pattern.fullmatch(normalized)
        for pattern in _GENERIC_ATTRIBUTION_PATTERNS
    )


def _is_likely_named_source(value: str) -> bool:
    """Keep strict checks for entity-shaped labels, not arbitrary clauses."""

    source = value.strip()
    if not source or _is_generic_attribution_source(source):
        return False
    return (
        len(source) <= 12
        or bool(re.search(r"[A-Z][A-Za-z0-9]", source))
        or "·" in source
        or source.endswith(_NAMED_SOURCE_SUFFIXES)
    )


def _attribution_source_identities(value: str) -> tuple[str, ...]:
    """Reduce a bounded attribution phrase to exact identities worth checking."""

    source = value.strip()
    if not source or _is_generic_attribution_source(source):
        return ()
    if source.startswith("来自") and source.endswith("的"):
        source = source[2:-1].strip()
    elif source.endswith("的"):
        source = source[:-1].strip()

    frequency_suffix_match = _ATTRIBUTION_FREQUENCY_SUFFIX_PATTERN.fullmatch(
        source
    )
    if frequency_suffix_match:
        source = frequency_suffix_match.group("source").strip()

    person_commentary_match = _PERSON_DATED_COMMENTARY_SOURCE_PATTERN.fullmatch(
        source
    )
    if person_commentary_match:
        return (person_commentary_match.group("person"),)

    latin_blog_match = _LATIN_BLOG_SOURCE_PATTERN.fullmatch(source)
    if latin_blog_match:
        return (latin_blog_match.group("organization").strip(),)

    named_vendor_match = _NAMED_VENDOR_ROLE_SOURCE_PATTERN.fullmatch(source)
    if named_vendor_match:
        return (named_vendor_match.group("organization").strip(),)

    latin_report_match = _LATIN_REPORT_SOURCE_PATTERN.fullmatch(source)
    if latin_report_match:
        return (latin_report_match.group("organization").strip(),)

    chinese_document_match = _CHINESE_DOCUMENT_SOURCE_PATTERN.fullmatch(source)
    if chinese_document_match:
        return (chinese_document_match.group("organization").strip(),)

    latin_role_match = _LATIN_ROLE_SUFFIX_SOURCE_PATTERN.fullmatch(source)
    if latin_role_match:
        return (latin_role_match.group("organization").strip(),)

    named_rule_match = _NAMED_RULE_ACTION_SOURCE_PATTERN.fullmatch(source)
    if named_rule_match:
        named_rule = named_rule_match.group("rule").strip()
        if named_rule and not _is_generic_attribution_source(named_rule):
            return (named_rule,)

    media_study_match = _MEDIA_BASED_STUDY_SOURCE_PATTERN.fullmatch(source)
    if media_study_match:
        return (media_study_match.group("outlet").strip(),)

    anonymous_speaker_match = _OUTLET_ANONYMOUS_SPEAKER_SOURCE_PATTERN.fullmatch(
        source
    )
    if anonymous_speaker_match:
        outlet = anonymous_speaker_match.group("outlet").strip()
        if outlet and not _is_generic_attribution_source(outlet):
            return (outlet,)

    school_group_match = _NAMED_SCHOOL_GROUP_SOURCE_PATTERN.fullmatch(source)
    if school_group_match:
        return (school_group_match.group("school").strip(),)

    surface_material_match = _NAMED_SURFACE_MATERIAL_SOURCE_PATTERN.fullmatch(source)
    if surface_material_match:
        named_source = surface_material_match.group("source").strip()
        if named_source and not _is_generic_attribution_source(named_source):
            return (named_source,)

    direct_composite_match = _DIRECT_COMPOSITE_NAMED_SOURCE_PATTERN.fullmatch(source)
    if direct_composite_match:
        identities = tuple(
            identity
            for identity in (
                direct_composite_match.group("outlet").strip(),
                direct_composite_match.group("speaker").strip(),
            )
            if identity and not _is_generic_attribution_source(identity)
        )
        if identities:
            return identities

    organization_action_match = (
        _RESEARCH_ORGANIZATION_ACTION_SOURCE_PATTERN.fullmatch(source)
    )
    if organization_action_match:
        return (organization_action_match.group("institution").strip(),)

    institution_person_match = _INSTITUTION_PERSON_SOURCE_PATTERN.fullmatch(source)
    if institution_person_match:
        return (
            institution_person_match.group("institution").strip(),
            institution_person_match.group("person").strip(),
        )

    for composite_pattern in (
        _MEDIA_RELAY_SOURCE_PATTERN,
        _MEDIA_REPORTS_DOCUMENT_SOURCE_PATTERN,
    ):
        composite_match = composite_pattern.fullmatch(source)
        if composite_match:
            identities: list[str] = []
            for component in (
                composite_match.group("outlet"),
                composite_match.group("underlying"),
            ):
                for identity in _attribution_source_identities(component):
                    if identity not in identities:
                        identities.append(identity)
            return tuple(identities)

    dated_survey_match = _DATED_SURVEY_ATTRIBUTION_SOURCE_PATTERN.fullmatch(source)
    if dated_survey_match:
        outlet = dated_survey_match.group("outlet").strip()
        if _is_likely_named_source(outlet):
            return (outlet,)

    if source.endswith("的研究"):
        source = source[: -len("的研究")].strip()
    source = re.sub(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?$", "", source).strip()
    if source.endswith("报道"):
        source = source[: -len("报道")].strip()
    if not source or _is_generic_attribution_source(source):
        return ()

    academic_match = _ACADEMIC_TEAM_SOURCE_PATTERN.fullmatch(source)
    if academic_match:
        people = tuple(
            item
            for item in re.split(r"[、，]", academic_match.group("people"))
            if item
        )
        return (academic_match.group("institution"), *people)

    person_match = _CONTEXTUAL_PERSON_SOURCE_PATTERN.fullmatch(source)
    if person_match:
        return (person_match.group("person"),)

    court_match = _COURT_DECISION_SOURCE_PATTERN.fullmatch(source)
    if court_match:
        return (court_match.group("court"),)

    action_match = _PERSON_ACTION_SOURCE_PATTERN.fullmatch(source)
    if action_match and not action_match.group("person").endswith(
        ("公司", "平台", "法院", "大学", "学院")
    ):
        return (action_match.group("person"),)

    latin_document_match = _LATIN_DOCUMENT_SOURCE_PATTERN.fullmatch(source)
    if latin_document_match:
        return (latin_document_match.group("organization").strip(),)

    for role_suffix in ("记者", "发言人", "研究员", "专家", "学者", "医生", "负责人"):
        if source.endswith(role_suffix):
            named_prefix = source[: -len(role_suffix)].strip()
            if named_prefix:
                source = named_prefix
            break
    if not source or _is_generic_attribution_source(source):
        return ()
    return (source,) if _is_likely_named_source(source) else ()


def _claim_attribution_sources(text: str) -> tuple[str, ...]:
    """Extract non-generic source identities from bounded attribution forms."""

    sources: list[str] = []
    specialized_spans: list[tuple[int, int]] = []

    def append_source(value: str) -> None:
        for source in _attribution_source_identities(value):
            if source not in sources:
                sources.append(source)

    for match in _BASED_ON_REPORT_ATTRIBUTION_PATTERN.finditer(text):
        specialized_spans.append(match.span())
        append_source(match.group("source"))
        append_source(match.group("underlying"))

    for match in _COMPOSITE_ATTRIBUTION_PATTERN.finditer(text):
        specialized_spans.append(match.span())
        append_source(match.group("outlet"))
        append_source(match.group("speaker"))

    for pattern in (
        _PUBLISHED_MATERIAL_ATTRIBUTION_PATTERN,
        _LEADING_SOURCE_VERB_PATTERN,
    ):
        for match in pattern.finditer(text):
            specialized_spans.append(match.span())
            append_source(match.group("source"))

    for pattern in _ATTRIBUTION_PATTERNS:
        for match in pattern.finditer(text):
            if any(
                match.start() < specialized_end and specialized_start < match.end()
                for specialized_start, specialized_end in specialized_spans
            ):
                continue
            append_source(match.group("source"))
    return tuple(sources)


def _is_obvious_user_generated_or_aggregation_source(
    *, title: str, url: str
) -> bool:
    """Identify a small set of explicit UGC/aggregation surfaces."""

    normalized_title = title.casefold()
    normalized_url = url.casefold()
    return (
        "财经头条" in normalized_title
        or "百家号" in normalized_title
        or "个人图书馆" in normalized_title
        or "cj.sina.cn/articles/view/" in normalized_url
        or "baijiahao.baidu.com/" in normalized_url
        or "360doc.com/" in normalized_url
    )


def _candidate_blocks(result_ref: str, content: str) -> tuple["_CandidateBlock", ...]:
    """Greedily pack exact sentence/line atoms into stable bounded blocks."""

    segments: list[str] = []
    start = 0
    for match in _BLOCK_BOUNDARY_PATTERN.finditer(content):
        segment = content[start : match.end()]
        start = match.end()
        if not segment:
            continue
        if not segment.strip() and segments:
            segments[-1] += segment
        else:
            segments.append(segment)
    if start < len(content):
        tail = content[start:]
        if not tail.strip() and segments:
            segments[-1] += tail
        elif tail:
            segments.append(tail)
    bounded_atoms = [
        segment[offset : offset + _MAX_BLOCK_LENGTH]
        for segment in segments
        for offset in range(0, len(segment), _MAX_BLOCK_LENGTH)
    ]
    packed_segments: list[str] = []
    pending = ""
    for atom in bounded_atoms:
        if pending and len(pending) + len(atom) > _MAX_BLOCK_LENGTH:
            packed_segments.append(pending)
            pending = ""
        pending += atom
    if pending:
        packed_segments.append(pending)
    return tuple(
        _CandidateBlock(
            block_id=f"{result_ref}-B{index:03d}",
            result_ref=result_ref,
            index=index,
            text=segment,
        )
        for index, segment in enumerate(packed_segments, start=1)
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    ref: str
    query: str
    result: SearchResult
    content: str


@dataclass(frozen=True, slots=True)
class _CandidateBlock:
    block_id: str
    result_ref: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class _EvidenceSelection:
    selection_ref: str
    result_ref: str
    excerpt: str
    source_type: EvidenceSourceType
    source_scope: str
    time_basis: str


@dataclass(frozen=True, slots=True)
class _ClaimSelection:
    text: str
    evidence_refs: tuple[str, ...]
    is_core: bool
    support_status: ClaimSupportStatus
    claim_kind: ClaimKind


@dataclass(frozen=True, slots=True)
class _TitleChainPart:
    component: str
    status: Literal["covered", "missing"]
    claim_numbers: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class _Assessment:
    status: _AssessmentStatus
    evidence: tuple[_EvidenceSelection, ...]
    claims: tuple[_ClaimSelection, ...]
    follow_up_queries: tuple[PlannedQuery, ...]
    blocking_gaps: tuple[str, ...]
    title_chain: tuple[_TitleChainPart, ...]


class ResearchGenerationError(RuntimeError):
    """Raised when query planning or evidence structuring repeatedly fails."""

    def __init__(
        self,
        message: str,
        *,
        llm_request_count: int = 0,
        search_request_count: int = 0,
        successful_search_count: int = 0,
        llm_usages: Sequence[LLMCallUsage] = (),
    ) -> None:
        super().__init__(message)
        self.llm_request_count = llm_request_count
        self.search_request_count = search_request_count
        self.successful_search_count = successful_search_count
        self.llm_usages = tuple(llm_usages)


class ResearchAgent:
    """Generate live queries and collect research material.

    ``collect_background`` is the production generation path: it performs live
    retrieval and returns compact source material without asking another model
    to prove a claim/evidence chain.  ``research`` retains the historical
    evidence-editor workflow so frozen experiments remain replayable.
    """

    _MAX_LLM_ATTEMPTS = 2
    _MAX_CANDIDATES = 20
    _MAX_EVIDENCE_ITEMS = 8
    _MAX_CLAIMS = 8
    _MIN_EVIDENCE_ITEMS = 2
    _MIN_SOURCE_DOMAINS = 2
    _MAX_EXCERPT_LENGTH = 1200
    _SOURCE_TYPES = frozenset(
        {
            "official_primary",
            "direct_terms",
            "primary_research",
            "authoritative_dataset",
            "reputable_reporting",
            "independent_secondary",
            "vendor_or_advocacy",
            "encyclopedia_social_personal",
        }
    )
    _CLAIM_KINDS = frozenset(
        {
            "rule_or_terms",
            "quantitative_state",
            "causal_effect",
            "case_event",
            "expert_opinion",
            "descriptive_context",
            "uncertainty_boundary",
        }
    )
    def __init__(
        self,
        llm: AsyncLLMClient,
        search: AsyncSearchProvider,
        *,
        config: ResearchConfig = ResearchConfig(),
    ) -> None:
        self._validate_config(config)
        self._llm = llm
        self._search = search
        self._config = config

    async def research(
        self,
        task: ScriptTask,
        *,
        current_date: date | None = None,
    ) -> ResearchOutcome:
        """Research one selected topic and return evidence or an explicit stop."""

        effective_date = current_date or datetime.now(_RESEARCH_TIMEZONE).date()
        research_date = effective_date.isoformat()
        llm_request_count = 0
        search_request_count = 0
        successful_search_count = 0
        usages: list[LLMCallUsage] = []
        logger.info("[1/5] 正在生成检索计划：%s", _log_text(task.topic))
        try:
            plan, plan_requests, plan_usages = await self._plan_queries(
                task,
                current_date=research_date,
            )
        except ResearchGenerationError as exc:
            raise self._accumulated_generation_error(
                exc,
                llm_request_count=llm_request_count,
                search_request_count=search_request_count,
                successful_search_count=successful_search_count,
                llm_usages=usages,
            ) from None
        llm_request_count += plan_requests
        usages.extend(plan_usages)
        queries = list(plan.queries)
        logger.info(
            "检索计划完成，共 %d 个初始查询：%s",
            len(queries),
            "；".join(_log_text(item.query) for item in queries),
        )
        logger.info("[2/5] 正在并发执行 %d 个初始搜索", len(queries))
        responses, errors = await self._search_queries(queries)
        search_request_count += len(queries)
        successful_search_count += len(responses)
        logger.info(
            "初始搜索完成：成功 %d，失败 %d",
            len(responses),
            len(errors),
        )

        candidates = self._collect_candidates(responses)
        if not candidates:
            logger.warning("[3/5] 没有可用搜索正文，调研停止")
            return self._outcome(
                status="insufficient_evidence",
                plan=plan,
                responses=responses,
                evidence=(),
                claims=(),
                errors=(*errors, "Search returned no usable evidence content."),
                llm_request_count=llm_request_count,
                search_request_count=search_request_count,
                executed_queries=queries,
                llm_usages=usages,
            )

        remaining_budget = self._config.max_search_requests - search_request_count
        logger.info(
            "[3/5] 正在从 %d 条候选结果中筛选证据和论断",
            len(candidates),
        )
        try:
            assessment, assessment_requests, assessment_usages = (
                await self._assess_evidence(
                    task,
                    plan,
                    candidates,
                    remaining_search_budget=remaining_budget,
                )
            )
        except ResearchGenerationError as exc:
            raise self._accumulated_generation_error(
                exc,
                llm_request_count=llm_request_count,
                search_request_count=search_request_count,
                successful_search_count=successful_search_count,
                llm_usages=usages,
            ) from None
        llm_request_count += assessment_requests
        usages.extend(assessment_usages)

        if assessment.status == "needs_more" and remaining_budget > 0:
            existing_queries = {item.query.casefold() for item in queries}
            follow_up_queries = tuple(
                item
                for item in assessment.follow_up_queries
                if item.query.casefold() not in existing_queries
            )[:remaining_budget]
            if follow_up_queries:
                logger.info(
                    "证据编辑要求补搜 %d 次：%s",
                    len(follow_up_queries),
                    "；".join(_log_text(item.query) for item in follow_up_queries),
                )
                follow_up_responses, follow_up_errors = await self._search_queries(
                    follow_up_queries
                )
                queries.extend(follow_up_queries)
                responses = (*responses, *follow_up_responses)
                errors = (*errors, *follow_up_errors)
                search_request_count += len(follow_up_queries)
                successful_search_count += len(follow_up_responses)
                candidates = self._collect_candidates(responses)
                logger.info(
                    "补充搜索完成：成功 %d，失败 %d；正在重新筛选证据",
                    len(follow_up_responses),
                    len(follow_up_errors),
                )
                try:
                    assessment, extra_requests, extra_usages = (
                        await self._assess_evidence(
                            task,
                            plan,
                            candidates,
                            remaining_search_budget=0,
                        )
                    )
                except ResearchGenerationError as exc:
                    raise self._accumulated_generation_error(
                        exc,
                        llm_request_count=llm_request_count,
                        search_request_count=search_request_count,
                        successful_search_count=successful_search_count,
                        llm_usages=usages,
                    ) from None
                llm_request_count += extra_requests
                usages.extend(extra_usages)

        evidence, claims = self._materialize(assessment, candidates)
        title_chain = tuple(
            TitleChainPart(
                component=part.component,  # type: ignore[arg-type]
                status=part.status,
                claim_ids=tuple(
                    claims[number - 1].claim_id for number in part.claim_numbers
                ),
                reason=part.reason,
            )
            for part in assessment.title_chain
        )
        status: Literal["ready", "insufficient_evidence"] = (
            "ready" if assessment.status == "ready" else "insufficient_evidence"
        )
        if status == "insufficient_evidence":
            errors = (
                *errors,
                *(f"Blocking evidence gap: {gap}" for gap in assessment.blocking_gaps),
            )
        quality_error = self._readiness_error(evidence, claims)
        if status == "ready" and quality_error is not None:
            status = "insufficient_evidence"
            errors = (*errors, quality_error)

        log_method = logger.info if status == "ready" else logger.warning
        log_method(
            "调研完成：status=%s，证据=%d，论断=%d，搜索请求=%d",
            status,
            len(evidence),
            len(claims),
            search_request_count,
        )

        return self._outcome(
            status=status,
            plan=plan,
            responses=responses,
            evidence=evidence,
            claims=claims,
            errors=errors,
            llm_request_count=llm_request_count,
            search_request_count=search_request_count,
            executed_queries=queries,
            llm_usages=usages,
            title_chain=title_chain,
        )

    async def collect_background(
        self,
        task: ScriptTask,
        *,
        current_date: date | None = None,
    ) -> ResearchOutcome:
        """Collect bounded live-search context and citation metadata.

        Search results are background for writing, not pre-approved factual
        claims.  No evidence-selection LLM call, title-chain audit, claim
        extraction, or grounding decision runs on this path.  The returned
        ``Evidence`` objects are reference records retained for offline scoring.
        """

        effective_date = current_date or datetime.now(_RESEARCH_TIMEZONE).date()
        research_date = effective_date.isoformat()
        logger.info("[1/3] 正在生成检索计划：%s", _log_text(task.topic))
        plan, plan_requests, plan_usages = await self._plan_queries(
            task,
            current_date=research_date,
        )
        queries = list(plan.queries)
        logger.info("[2/3] 正在并发执行 %d 个背景搜索", len(queries))
        responses, errors = await self._search_queries(queries)
        candidates = self._collect_candidates(responses)
        references = self._background_references(candidates)
        status: Literal["ready", "insufficient_evidence"] = (
            "ready" if references else "insufficient_evidence"
        )
        if not references:
            errors = (*errors, "Search returned no usable background content.")
            logger.warning("[3/3] 没有可用背景资料，生成停止")
        else:
            logger.info(
                "[3/3] 背景资料就绪：references=%d，搜索成功=%d/%d",
                len(references),
                len(responses),
                len(queries),
            )
        return ResearchOutcome(
            status=status,
            query_plan=plan,
            search_responses=tuple(responses),
            evidence=references,
            claims=(),
            errors=tuple(errors),
            query_plan_prompt_version=RESEARCH_QUERY_PLAN_PROMPT_VERSION,
            evidence_prompt_version=BACKGROUND_SELECTION_VERSION,
            llm_request_count=plan_requests,
            search_request_count=len(queries),
            executed_queries=tuple(queries),
            llm_usages=plan_usages,
            title_chain=(),
        )

    async def retry_failed_background_searches(
        self,
        frozen: ResearchOutcome,
    ) -> ResearchOutcome:
        """Retry only failed searches from one frozen background collection.

        The Hy3 query plan is never regenerated. Queries that already produced
        a response are not sent again, and the original failures remain in the
        returned audit history. This recovery path is for transport/provider
        failures, not for expanding a usable background with new queries.
        """

        if frozen.evidence or frozen.claims or frozen.title_chain:
            raise ValueError(
                "Background search recovery requires an empty failed snapshot."
            )
        if frozen.status != "insufficient_evidence":
            raise ValueError(
                "Background search recovery requires insufficient_evidence status."
            )
        planned_queries = frozen.executed_queries or frozen.query_plan.queries
        completed_queries = {
            response.query.casefold() for response in frozen.search_responses
        }
        failed_queries = tuple(
            item
            for item in planned_queries
            if item.query.casefold() not in completed_queries
        )
        if not failed_queries:
            raise ValueError("Background snapshot contains no failed search to retry.")

        retry_responses, retry_errors = await self._search_queries(failed_queries)
        responses = (*frozen.search_responses, *retry_responses)
        candidates = self._collect_candidates(responses)
        references = self._background_references(candidates)
        status: Literal["ready", "insufficient_evidence"] = (
            "ready" if references else "insufficient_evidence"
        )
        recovery_note = (
            f"Operational recovery retried {len(failed_queries)} failed frozen "
            "background queries without regenerating the query plan."
        )
        errors = (*frozen.errors, recovery_note, *retry_errors)
        if not references:
            errors = (*errors, "Recovery returned no usable background content.")
        return ResearchOutcome(
            status=status,
            query_plan=frozen.query_plan,
            search_responses=tuple(responses),
            evidence=references,
            claims=(),
            errors=tuple(errors),
            query_plan_prompt_version=frozen.query_plan_prompt_version,
            evidence_prompt_version=frozen.evidence_prompt_version,
            llm_request_count=frozen.llm_request_count,
            search_request_count=(
                frozen.search_request_count + len(failed_queries)
            ),
            executed_queries=tuple(planned_queries),
            llm_usages=frozen.llm_usages,
            title_chain=(),
        )

    def _background_references(
        self,
        candidates: Sequence[_Candidate],
    ) -> tuple[Evidence, ...]:
        """Turn top interleaved search results into compact reference records."""

        return tuple(
            Evidence(
                evidence_id=f"E{index:03d}",
                result_ref=item.ref,
                title=item.result.title,
                url=item.result.url,
                excerpt=item.content[: self._MAX_EXCERPT_LENGTH],
                source_query=item.query,
                published_at=item.result.published_at,
                content_hash=item.result.content_hash,
                score=item.result.score,
            )
            for index, item in enumerate(
                candidates[: self._MAX_EVIDENCE_ITEMS],
                start=1,
            )
        )

    async def _plan_queries(
        self,
        task: ScriptTask,
        *,
        current_date: str,
    ) -> tuple[QueryPlan, int, tuple[LLMCallUsage, ...]]:
        schema = {
            "goal": "本次调研要回答的核心问题",
            "must_verify": ["必须核实的信息点"],
            "queries": [
                {
                    "query": "可直接交给搜索服务的查询词",
                    "purpose": "该查询要解决的信息缺口",
                }
            ],
        }
        input_payload = {
            "current_date": current_date,
            "task": asdict(task),
        }
        prompt = (
            f"生成恰好 {self._config.initial_query_count} 个互补查询。查询之间不得只是换词，"
            "至少一个查询优先寻找原始或权威来源。若题目涉及法律、医疗、金融、平台规则或公共"
            "政策，至少一个查询应直接面向制定者的现行原文及例外或试点；若题目涉及效果、因果"
            "或预测能力，至少一个查询应面向原始研究的方法、样本和局限，不能只搜厂商宣传。"
            "查询必须覆盖回答标题所需的完整决策链，而不是用容易搜索的规模或采用率替代价格、"
            "效果、公平性等真正问题。current_date 是本次运行的真实日期；涉及当前"
            "状态或近期进展时以它为时间锚点，不得把更早年份误写成“近期”。只有核实历史起点时"
            "才使用旧年份，并同时保留至少一个覆盖最新进展的查询。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 是本次任务数据：\n"
            f"{json.dumps(input_payload, ensure_ascii=False)}"
        )
        messages = (
            ChatMessage(role="system", content=RESEARCH_QUERY_PLAN_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )
        return await self._request_with_retry(
            messages,
            lambda response: self._parse_plan(response, current_date=current_date),
            stage="query planning",
            usage_stage="research.query_plan",
        )

    def _parse_plan(self, response: str, *, current_date: str) -> QueryPlan:
        payload = json_object(response)
        goal = required_text(payload, "goal", max_length=300)
        must_verify = text_list(
            payload,
            "must_verify",
            minimum=1,
            maximum=10,
            item_max_length=160,
        )
        raw_queries = payload.get("queries")
        if (
            not isinstance(raw_queries, list)
            or len(raw_queries) != self._config.initial_query_count
        ):
            raise StructuredOutputError("Response contains an invalid query count.")
        queries: list[PlannedQuery] = []
        seen: set[str] = set()
        for raw_query in raw_queries:
            if not isinstance(raw_query, dict):
                raise StructuredOutputError("Response contains an invalid query.")
            query = required_text(raw_query, "query", max_length=180)
            purpose = required_text(raw_query, "purpose", max_length=200)
            normalized = query.casefold()
            if normalized in seen:
                raise StructuredOutputError("Response contains duplicate queries.")
            seen.add(normalized)
            queries.append(PlannedQuery(query=query, purpose=purpose))
        return QueryPlan(
            goal=goal,
            must_verify=must_verify,
            queries=tuple(queries),
            current_date=current_date,
        )

    async def _search_queries(
        self,
        queries: Sequence[PlannedQuery],
    ) -> tuple[tuple[SearchResponse, ...], tuple[str, ...]]:
        semaphore = asyncio.Semaphore(self._config.max_search_concurrency)
        query_count = len(queries)

        async def run(index: int, item: PlannedQuery) -> tuple[SearchResponse | None, str | None]:
            async with semaphore:
                logger.info(
                    "搜索 %d/%d 开始：%s",
                    index,
                    query_count,
                    _log_text(item.query),
                )
                try:
                    response = await self._search.search(
                        item.query,
                        limit=self._config.results_per_query,
                    )
                except SearchProviderError:
                    logger.warning(
                        "搜索 %d/%d 失败：%s",
                        index,
                        query_count,
                        _log_text(item.query),
                    )
                    return None, f"Search request {index} failed."
                logger.info(
                    "搜索 %d/%d 完成：返回 %d 条结果",
                    index,
                    query_count,
                    len(response.results),
                )
                return response, None

        results = await asyncio.gather(
            *(run(index, item) for index, item in enumerate(queries, start=1))
        )
        return (
            tuple(response for response, _ in results if response is not None),
            tuple(error for _, error in results if error is not None),
        )

    def _collect_candidates(
        self,
        responses: Sequence[SearchResponse],
    ) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        seen_urls: set[str] = set()
        max_result_count = max((len(response.results) for response in responses), default=0)
        for result_index in range(max_result_count):
            for response in responses:
                if result_index >= len(response.results):
                    continue
                result = response.results[result_index]
                normalized_url = result.url.strip()
                parsed = urlparse(normalized_url)
                if (
                    not normalized_url
                    or normalized_url in seen_urls
                    or parsed.scheme not in {"http", "https"}
                    or not parsed.netloc
                ):
                    continue
                content = (result.raw_content or result.snippet).strip()
                if not content:
                    continue
                seen_urls.add(normalized_url)
                candidates.append(
                    _Candidate(
                        ref=f"R{len(candidates) + 1:03d}",
                        query=response.query,
                        result=result,
                        content=content[: self._config.max_content_chars_per_result],
                    )
                )
                if len(candidates) >= self._MAX_CANDIDATES:
                    return tuple(candidates)
        return tuple(candidates)

    async def _assess_evidence(
        self,
        task: ScriptTask,
        plan: QueryPlan,
        candidates: Sequence[_Candidate],
        *,
        remaining_search_budget: int,
    ) -> tuple[_Assessment, int, tuple[LLMCallUsage, ...]]:
        schema = {
            "status": "ready | needs_more | insufficient_evidence",
            "evidence": [
                {
                    "selection_ref": "S001",
                    "result_ref": "R001",
                    "block_ids": ["R001-B001", "R001-B002"],
                    "source_type": (
                        "official_primary | direct_terms | primary_research | "
                        "authoritative_dataset | reputable_reporting | "
                        "independent_secondary | vendor_or_advocacy | "
                        "encyclopedia_social_personal"
                    ),
                    "source_scope": "来源直接覆盖的地区、对象、样本或产品范围",
                    "time_basis": "来源明确给出的发布日期、生效时间或数据期间；没有则写 unknown",
                }
            ],
            "claims": [
                {
                    "text": "证据可以支持的候选论断（不超过300个字符）",
                    "evidence_refs": ["S001"],
                    "is_core": True,
                    "support_status": "supported | conflicting | unsupported",
                    "claim_kind": (
                        "rule_or_terms | quantitative_state | causal_effect | "
                        "case_event | expert_opinion | descriptive_context | "
                        "uncertainty_boundary"
                    ),
                }
            ],
            "title_chain": {
                "subject_scope": {
                    "status": "covered | missing",
                    "claim_numbers": [1],
                    "reason": "为何这些 core claim 直接覆盖标题主体和范围",
                },
                "stated_context": {
                    "status": "covered | missing",
                    "claim_numbers": [1],
                    "reason": "为何直接覆盖题设情境；无独立情境时说明与主体合并",
                },
                "question_predicate": {
                    "status": "covered | missing",
                    "claim_numbers": [1],
                    "reason": "为何直接回答标题所问效果、因果、利弊或决策变量",
                },
            },
            "follow_up_queries": [
                {
                    "query": "补充查询",
                    "purpose": "仍需补足的信息",
                }
            ],
            "blocking_gaps": ["缺失信息，以及它为何会实质改变核心结论"],
        }
        blocks_by_result = {
            item.ref: _candidate_blocks(item.ref, item.content) for item in candidates
        }
        input_payload = {
            "task": asdict(task),
            "current_date": plan.current_date,
            "research_goal": plan.goal,
            "must_verify": plan.must_verify,
            "remaining_search_budget": remaining_search_budget,
            "candidates": [
                {
                    "result_ref": item.ref,
                    "query": item.query,
                    "title": item.result.title,
                    "url": item.result.url,
                    "published_at": item.result.published_at,
                    "score": item.result.score,
                    "blocks": [
                        {"block_id": block.block_id, "text": block.text}
                        for block in blocks_by_result[item.ref]
                    ],
                }
                for item in candidates
            ],
        }
        max_core_claims = self._max_core_claims(task.target_length)
        prompt = (
            f"最多选择 {self._MAX_EVIDENCE_ITEMS} 条证据和 {self._MAX_CLAIMS} 条候选论断。"
            f"本任务正文目标为 {task.target_length} 个非空白字符，最多标记 {max_core_claims} 条"
            "is_core=true 的主干论断；其余有用材料标为非核心，避免短稿被事实清单挤满。"
            "长稿中准备写入的日期、症状、流程、样本、规则边界等必须各自进入 supported atomic "
            "claim；不能指望写作阶段直接从 excerpt 扩写。若 claims 加上不增加新事实的必要解释"
            "不足以达到目标字数允许下限，应补充细粒度 claim 或返回非 ready。"
            "每个 evidence.selection_ref 必须唯一。每条 evidence 只能选择 1 至 3 个属于"
            "同一 result_ref、按原文顺序连续的 block_ids，拼接后不得超过 1200 个字符。"
            "block_id 必须从候选 blocks 中逐字复制，不得根据编号规律推算；需要引用同一来源"
            "中彼此分离的片段时，必须拆成不同 selection_ref。不得输出或手抄 excerpt；程序会"
            "用 block_ids 从原 content 精确回填。完全相同的"
            "result_ref 与 block_ids 组合不得重复；claim 只能通过 evidence_refs 引用已选择的"
            "selection_ref。"
            "剩余搜索预算不是必须用完的配额：已有材料足以支持克制且有边界的核心结论时必须返回"
            "ready。needs_more 只能在某项缺失信息会实质改变核心结论、且现有 candidates 无法支持"
            "该信息时使用；每个补充查询的 purpose 必须明确缺少什么以及为何现有材料不够。"
            "follow_up_queries 不得重复已有查询，也不得超过剩余预算。ready 时 blocking_gaps 必须"
            "为空；needs_more 或 insufficient_evidence 时必须逐项写出真正阻断核心回答的缺口，"
            "blocking_gaps 必须是最多4个非空字符串组成的 JSON 数组，不能返回对象或超过4项。"
            "不能把非必要的当年事故、品牌清单或补充数字列为阻断项。每条 claim.text 不得超过"
            "300个字符，且只写一个主体的一项事实、规则、措施、状态或效果；多个平台、地区、"
            "状态或“措施＋效果”必须拆开。同一报道中的不同消费者、患者或案例也必须拆开，"
            "不得把甲的默认开通多扣款与乙的借款违约金合成一个案例或因果。"
            "核心证据摘录必须自包含，不能用缺少先行词的‘这项概念’‘他们’等句子。题目所需的"
            "因果桥、统计分母、适用地区、普通人影响、现行规则和试点例外若未被候选材料直接支持，"
            "不得靠常识补齐；它会实质改变结论时应 needs_more 或 insufficient_evidence。关键数字、"
            "规则或效果只有厂商、媒体、百科、社交帖子等材料时，在 claim 中明确归因，不能写成"
            "独立共识。source_type 按来源本身而非文章声称的对象分类：政府、法院、国际组织原文"
            "为 official_primary；银行或平台自己的现行条款为 direct_terms；原始论文为 "
            "primary_research；官方统计或正式监测数据为 authoritative_dataset；专业媒体对事件的"
            "直接报道为 reputable_reporting；评论和二手分析为 independent_secondary；厂商宣传、"
            "倡议材料为 vendor_or_advocacy；百科、社交帖子、个人网站或体验文为 "
            "encyclopedia_social_personal。转载官方文字仍按转载来源分类，不能冒充原文。"
            "标题带‘财经头条’‘百家号’‘个人图书馆’等明确用户发布或聚合标识的页面，不是"
            "专业媒体直接报道，不能标为 reputable_reporting。未署名的‘有评论’或二手观点不能"
            "作为覆盖 question_predicate 的核心 expert_opinion；必须补搜具名且可核验的专业来源，"
            "否则停止。PDF或论文外观也不自动证明因果，仍须由 excerpt 的方法、样本和结论直接"
            "支持 causal_effect。"
            "ready 时 core claim 集合必须共同覆盖标题的对象与谓词/决策变量，每条 core claim"
            "至少推进一个变量，不得用相邻指标替代。claim_kind 重叠时按 rule_or_terms > "
            "causal_effect > quantitative_state > case_event > expert_opinion > "
            "descriptive_context 的优先级分类。法律或规则上的权利、义务、"
            "权限、合法性或适用边界，无论来自官方原文还是媒体/评论，都必须标为 rule_or_terms；"
            "媒体/评论不因此变成规则原文。如果只有这类二手来源，该论断不能 is_core=true，"
            "必须补搜 official_primary/direct_terms 或在预算耗尽时停止。不得把相关性标成 "
            "causal_effect。每条 core claim 必须至少引用一种与其 claim_kind 匹配的来源：规则或条款"
            "只能用 official_primary/direct_terms，数量状态和因果效果只能用官方原文、原始研究或"
            "权威数据；事件可用官方原文或专业媒体。这些 source_type 门槛不能替代所引原文块对"
            "claim 的直接蕴含。source_scope/time_basis 是内部边界，claim 不得违反或放大；只在"
            "它们实质限制结论时自然交代，time_basis=unknown 不得写入 claim。具名归因的名称只能"
            "逐字取自引用证据的 title 或程序回填 excerpt；URL 只用于检查主体冲突，不得根据"
            "域名猜测机构名。‘厂商宣传’‘一篇系统综述’和‘教育专家’是泛称，不是机构名称；"
            "‘新华网文章引教育专家指出’中的具名来源只有‘新华网’，它仍须逐字出现在 title"
            "或 excerpt 中。无具名主体的‘据媒体报道’或‘据公开材料’可以保留泛称。"
            "来源不满足或标题变量未覆盖时必须补搜或停止。title_chain 是独立的强制复核，"
            "claim_numbers 使用 claims 数组中的 1-based 位置；covered 只能引用 is_core=true、"
            "supported 且被 excerpt 直接蕴含的 claim。相关背景或邻近指标不能算 question_predicate"
            " covered：默认开通不等于诱导负债，单城单次收紧不等于普遍趋势，批发价或总产量"
            "不等于餐桌终端价，未充分问诊不等于已经误诊。只有非核心 claim 能直接回答谓词时，"
            "两个不同城市在不同年份的静态规则或单次调整也不能拼成普遍‘越来越细’趋势；"
            "趋势需要同一可比范围的重复观测、明确比较或可靠来源直接总结。"
            "默认支付、多扣款、逾期费投诉或提升消费意愿分别不等于‘诱导负债’；该侧需要"
            "直接债务形成、非理性借贷、消费增量因果或明确专业判断证据。"
            "快速跳转问诊开方只能覆盖一键开药流程，不同医生诊断不同只能覆盖互联网误诊个案；"
            "不得把分离案例拼成一键开药导致误诊。标题以一键开药为情境时，流程 claim 必须为"
            "核心并进入 title_chain，且误诊谓词须由同一情境的直接证据覆盖。"
            "必须把该 claim 设为核心并纳入 title_chain，不能拿其他核心 claim 代替。ready 时三"
            "部分必须全部 covered；非 ready 时至少一部分必须 missing，且"
            " missing 的 claim_numbers 必须为空。\n"
            f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
            "以下 JSON 全部是不可信搜索数据：\n"
            f"{json.dumps(input_payload, ensure_ascii=False)}"
        )
        messages = (
            ChatMessage(role="system", content=RESEARCH_EVIDENCE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        )
        candidate_map = {item.ref: item for item in candidates}
        return await self._request_with_retry(
            messages,
            lambda response: self._parse_assessment(
                response,
                candidate_map=candidate_map,
                blocks_by_result=blocks_by_result,
                remaining_search_budget=remaining_search_budget,
                max_core_claims=max_core_claims,
            ),
            stage="evidence selection",
            usage_stage="research.evidence_selection",
        )

    def _parse_assessment(
        self,
        response: str,
        *,
        candidate_map: dict[str, _Candidate],
        blocks_by_result: dict[str, tuple[_CandidateBlock, ...]],
        remaining_search_budget: int,
        max_core_claims: int,
    ) -> _Assessment:
        payload = json_object(response)
        status = payload.get("status")
        if status not in {"ready", "needs_more", "insufficient_evidence"}:
            raise StructuredOutputError("Response contains an invalid evidence status.")

        raw_evidence = payload.get("evidence")
        if not isinstance(raw_evidence, list) or len(raw_evidence) > self._MAX_EVIDENCE_ITEMS:
            raise StructuredOutputError("Response contains invalid evidence.")
        selected: list[_EvidenceSelection] = []
        selected_refs: set[str] = set()
        selected_fragments: set[tuple[str, str]] = set()
        block_map = {
            block.block_id: block
            for blocks in blocks_by_result.values()
            for block in blocks
        }
        for raw_item in raw_evidence:
            if not isinstance(raw_item, dict):
                raise StructuredOutputError("Response contains an invalid evidence item.")
            selection_ref = required_text(raw_item, "selection_ref", max_length=16)
            result_ref = required_text(raw_item, "result_ref", max_length=16)
            candidate = candidate_map.get(result_ref)
            if candidate is None:
                raise StructuredOutputError("Response references an unknown search result.")
            raw_block_ids = raw_item.get("block_ids")
            if (
                not isinstance(raw_block_ids, list)
                or not 1 <= len(raw_block_ids) <= 3
                or any(
                    not isinstance(block_id, str) or not block_id.strip()
                    for block_id in raw_block_ids
                )
            ):
                raise StructuredOutputError(
                    "Evidence block_ids must contain one to three IDs."
                )
            block_ids = tuple(block_id.strip() for block_id in raw_block_ids)
            selected_blocks: list[_CandidateBlock] = []
            for block_id in block_ids:
                block = block_map.get(block_id)
                if block is None:
                    available_blocks = blocks_by_result[result_ref]
                    available_range = (
                        f"{available_blocks[0].block_id} through "
                        f"{available_blocks[-1].block_id}"
                    )
                    raise StructuredOutputError(
                        f"Evidence references an unknown block_id: {block_id}. "
                        f"Valid block_ids for {result_ref} run from {available_range}; "
                        "copy an ID exactly from the candidate blocks."
                    )
                if block.result_ref != result_ref:
                    raise StructuredOutputError(
                        f"Evidence block_id {block_id} belongs to {block.result_ref}, "
                        f"not {result_ref}; all block_ids must belong to its result_ref."
                    )
                selected_blocks.append(block)
            block_indices = [block.index for block in selected_blocks]
            if block_indices != list(
                range(block_indices[0], block_indices[0] + len(block_indices))
            ):
                raise StructuredOutputError(
                    f"Evidence block_ids for {result_ref} must be in source order and "
                    f"contiguous; received {list(block_ids)}. If the passages are "
                    "separated, create separate evidence items with different "
                    "selection_ref values."
                )
            excerpt = "".join(block.text for block in selected_blocks)
            if len(excerpt) > self._MAX_EXCERPT_LENGTH:
                raise StructuredOutputError(
                    "Evidence blocks exceed the 1200-character excerpt limit."
                )
            source_type = required_text(
                raw_item,
                "source_type",
                max_length=40,
            )
            if source_type not in self._SOURCE_TYPES:
                raise StructuredOutputError("Evidence source_type is invalid.")
            candidate = candidate_map[result_ref]
            if source_type in {
                "official_primary",
                "direct_terms",
                "primary_research",
                "authoritative_dataset",
                "reputable_reporting",
            } and (
                _is_obvious_user_generated_or_aggregation_source(
                    title=candidate.result.title,
                    url=candidate.result.url,
                )
            ):
                raise StructuredOutputError(
                    "Evidence high-trust source_type conflicts with an obvious "
                    "user-generated or aggregation surface."
                )
            source_scope = required_text(
                raw_item,
                "source_scope",
                max_length=240,
            )
            time_basis = required_text(
                raw_item,
                "time_basis",
                max_length=160,
            )
            if selection_ref in selected_refs:
                raise StructuredOutputError("Response contains duplicate evidence refs.")
            fragment_key = (result_ref, excerpt)
            if fragment_key in selected_fragments:
                raise StructuredOutputError("Response selects an identical evidence fragment.")
            selected_refs.add(selection_ref)
            selected_fragments.add(fragment_key)
            selected.append(
                _EvidenceSelection(
                    selection_ref=selection_ref,
                    result_ref=result_ref,
                    excerpt=excerpt,
                    source_type=source_type,  # type: ignore[arg-type]
                    source_scope=source_scope,
                    time_basis=time_basis,
                )
            )

        evidence_by_ref = {item.selection_ref: item for item in selected}
        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list) or len(raw_claims) > self._MAX_CLAIMS:
            raise StructuredOutputError("Response contains invalid claims.")
        claims: list[_ClaimSelection] = []
        seen_claims: set[str] = set()
        for claim_index, raw_claim in enumerate(raw_claims):
            if not isinstance(raw_claim, dict):
                raise StructuredOutputError("Response contains an invalid claim.")
            raw_text = raw_claim.get("text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise StructuredOutputError(
                    f"Response is missing claims[{claim_index}].text."
                )
            text = raw_text.strip()
            if len(text) > 300:
                raise StructuredOutputError(
                    f"Response claims[{claim_index}].text has {len(text)} characters; "
                    "the maximum is 300."
                )
            normalized_text = text.casefold()
            if normalized_text in seen_claims:
                raise StructuredOutputError("Response contains duplicate claims.")
            seen_claims.add(normalized_text)
            raw_refs = raw_claim.get("evidence_refs")
            if not isinstance(raw_refs, list) or not raw_refs:
                raise StructuredOutputError("Response contains a claim without refs.")
            evidence_refs = tuple(
                dict.fromkeys(
                    ref.strip()
                    for ref in raw_refs
                    if isinstance(ref, str) and ref.strip()
                )
            )
            if not evidence_refs or any(ref not in selected_refs for ref in evidence_refs):
                raise StructuredOutputError("Claim references unselected evidence.")
            is_core = raw_claim.get("is_core")
            if not isinstance(is_core, bool):
                raise StructuredOutputError("Claim is_core must be boolean.")
            support_status = raw_claim.get("support_status")
            if support_status not in {"supported", "conflicting", "unsupported"}:
                raise StructuredOutputError("Claim support_status is invalid.")
            claim_kind = required_text(raw_claim, "claim_kind", max_length=40)
            if claim_kind not in self._CLAIM_KINDS:
                raise StructuredOutputError("Claim claim_kind is invalid.")
            for attributed_source in _claim_attribution_sources(text):
                source_identity = _normalized_source_identity(attributed_source)
                referenced_source_fields: list[str] = []
                for evidence_ref in evidence_refs:
                    evidence_item = evidence_by_ref[evidence_ref]
                    candidate = candidate_map[evidence_item.result_ref]
                    referenced_source_fields.extend(
                        (candidate.result.title, evidence_item.excerpt)
                    )
                if not any(
                    source_identity in _normalized_source_identity(field)
                    for field in referenced_source_fields
                ):
                    raise StructuredOutputError(
                        "Claim names an attribution source absent from its referenced "
                        f"evidence: {attributed_source}."
                    )
            claims.append(
                _ClaimSelection(
                    text=text,
                    evidence_refs=evidence_refs,
                    is_core=is_core,
                    support_status=support_status,
                    claim_kind=claim_kind,  # type: ignore[arg-type]
                )
            )

        title_chain_components = (
            "subject_scope",
            "stated_context",
            "question_predicate",
        )
        raw_title_chain = payload.get("title_chain")
        if (
            not isinstance(raw_title_chain, dict)
            or set(raw_title_chain) != set(title_chain_components)
        ):
            raise StructuredOutputError(
                "Response title_chain must contain subject_scope, stated_context, "
                "and question_predicate."
            )
        title_chain: list[_TitleChainPart] = []
        for component in title_chain_components:
            raw_part = raw_title_chain[component]
            if not isinstance(raw_part, dict) or set(raw_part) != {
                "status",
                "claim_numbers",
                "reason",
            }:
                raise StructuredOutputError(
                    f"Response title_chain.{component} is invalid."
                )
            coverage_status = required_text(raw_part, "status", max_length=16)
            if coverage_status not in {"covered", "missing"}:
                raise StructuredOutputError(
                    f"Response title_chain.{component}.status is invalid."
                )
            raw_claim_numbers = raw_part.get("claim_numbers")
            if not isinstance(raw_claim_numbers, list):
                raise StructuredOutputError(
                    f"Response title_chain.{component}.claim_numbers must be a list."
                )
            claim_numbers: list[int] = []
            for claim_number in raw_claim_numbers:
                if (
                    isinstance(claim_number, bool)
                    or not isinstance(claim_number, int)
                    or not 1 <= claim_number <= len(claims)
                    or claim_number in claim_numbers
                ):
                    raise StructuredOutputError(
                        f"Response title_chain.{component} contains an invalid claim number."
                    )
                claim_numbers.append(claim_number)
            reason = required_text(raw_part, "reason", max_length=300)
            if coverage_status == "covered":
                if not claim_numbers or any(
                    not claims[number - 1].is_core
                    or claims[number - 1].support_status != "supported"
                    for number in claim_numbers
                ):
                    raise StructuredOutputError(
                        f"Covered title_chain.{component} requires supported core claims."
                    )
            elif claim_numbers:
                raise StructuredOutputError(
                    f"Missing title_chain.{component} must not reference claims."
                )
            title_chain.append(
                _TitleChainPart(
                    component=component,
                    status=coverage_status,  # type: ignore[arg-type]
                    claim_numbers=tuple(claim_numbers),
                    reason=reason,
                )
            )

        raw_follow_ups = payload.get("follow_up_queries", [])
        if not isinstance(raw_follow_ups, list):
            raise StructuredOutputError("Response contains invalid follow-up queries.")
        if len(raw_follow_ups) > remaining_search_budget:
            raise StructuredOutputError("Response exceeds the remaining search budget.")
        follow_ups: list[PlannedQuery] = []
        seen_follow_ups: set[str] = set()
        for raw_query in raw_follow_ups:
            if not isinstance(raw_query, dict):
                raise StructuredOutputError("Response contains an invalid follow-up query.")
            query = required_text(raw_query, "query", max_length=180)
            purpose = required_text(raw_query, "purpose", max_length=200)
            normalized_query = query.casefold()
            if normalized_query in seen_follow_ups:
                raise StructuredOutputError("Response contains duplicate follow-up queries.")
            seen_follow_ups.add(normalized_query)
            follow_ups.append(PlannedQuery(query=query, purpose=purpose))

        blocking_gaps = text_list(
            {"blocking_gaps": payload.get("blocking_gaps", [])},
            "blocking_gaps",
            minimum=0,
            maximum=4,
            item_max_length=240,
        )
        if status == "ready" and blocking_gaps:
            raise StructuredOutputError("ready must not contain blocking evidence gaps.")
        if status != "ready" and not blocking_gaps:
            raise StructuredOutputError(
                "Non-ready evidence status requires a concrete blocking gap."
            )

        if status == "needs_more" and (remaining_search_budget < 1 or not follow_ups):
            raise StructuredOutputError("needs_more requires a follow-up query and budget.")
        covered_components = {
            part.component for part in title_chain if part.status == "covered"
        }
        if status == "ready" and covered_components != set(title_chain_components):
            raise StructuredOutputError(
                "ready requires every title_chain component to be covered."
            )
        if status != "ready" and covered_components == set(title_chain_components):
            raise StructuredOutputError(
                "Non-ready evidence status requires a missing title_chain component."
            )
        core_claims = [item for item in claims if item.is_core]
        if len(core_claims) > max_core_claims:
            raise StructuredOutputError(
                f"Response contains more than {max_core_claims} core claims for this "
                "target length."
            )
        if status == "ready":
            for core_claim in core_claims:
                source_types = {
                    evidence_by_ref[reference].source_type
                    for reference in core_claim.evidence_refs
                }
                allowed_source_types = CORE_SOURCE_TYPES_BY_CLAIM_KIND[
                    core_claim.claim_kind
                ]
                if source_types.isdisjoint(allowed_source_types):
                    raise StructuredOutputError(
                        "Core claim source quality does not satisfy its claim_kind."
                    )
        if status == "ready" and (
            not selected
            or not claims
            or not core_claims
            or any(item.support_status != "supported" for item in core_claims)
        ):
            raise StructuredOutputError(
                "ready requires every core claim to be supported."
            )
        return _Assessment(
            status=status,
            evidence=tuple(selected),
            claims=tuple(claims),
            follow_up_queries=tuple(follow_ups),
            blocking_gaps=blocking_gaps,
            title_chain=tuple(title_chain),
        )

    @staticmethod
    def _max_core_claims(target_length: int) -> int:
        """Bound must-use claims so evidence density fits the requested script."""

        if target_length <= 320:
            return 2
        if target_length <= 550:
            return 3
        return 4

    @staticmethod
    def _materialize(
        assessment: _Assessment,
        candidates: Sequence[_Candidate],
    ) -> tuple[tuple[Evidence, ...], tuple[Claim, ...]]:
        candidate_map = {item.ref: item for item in candidates}
        ref_to_evidence_id = {
            item.selection_ref: f"E{index:03d}"
            for index, item in enumerate(assessment.evidence, start=1)
        }
        evidence = tuple(
            Evidence(
                evidence_id=ref_to_evidence_id[item.selection_ref],
                result_ref=item.result_ref,
                title=candidate_map[item.result_ref].result.title,
                url=candidate_map[item.result_ref].result.url,
                excerpt=item.excerpt,
                source_query=candidate_map[item.result_ref].query,
                published_at=candidate_map[item.result_ref].result.published_at,
                content_hash=candidate_map[item.result_ref].result.content_hash,
                score=candidate_map[item.result_ref].result.score,
                source_type=item.source_type,
                source_scope=item.source_scope,
                time_basis=item.time_basis,
            )
            for item in assessment.evidence
        )
        claims = tuple(
            Claim(
                claim_id=f"C{index:03d}",
                text=item.text,
                evidence_ids=tuple(
                    ref_to_evidence_id[ref]
                    for ref in item.evidence_refs
                ),
                is_core=item.is_core,
                support_status=item.support_status,
                claim_kind=item.claim_kind,
            )
            for index, item in enumerate(assessment.claims, start=1)
        )
        return evidence, claims

    def _readiness_error(
        self,
        evidence: Sequence[Evidence],
        claims: Sequence[Claim],
    ) -> str | None:
        if len(evidence) < self._MIN_EVIDENCE_ITEMS:
            return "Ready research requires at least two evidence items."
        domains = {
            urlparse(item.url).netloc.casefold().removeprefix("www.")
            for item in evidence
        }
        if len(domains) < self._MIN_SOURCE_DOMAINS:
            return "Ready research requires evidence from at least two source domains."
        core_claims = [item for item in claims if item.is_core]
        if not core_claims or any(
            item.support_status != "supported" for item in core_claims
        ):
            return "Every core claim must be supported before script generation."
        return None

    async def _request_with_retry(
        self,
        messages: Sequence[ChatMessage],
        parser: Any,
        *,
        stage: str,
        usage_stage: str,
    ) -> tuple[Any, int, tuple[LLMCallUsage, ...]]:
        current_messages = tuple(messages)
        usages: list[LLMCallUsage] = []
        last_failure = "unknown validation failure"
        request_count = 0
        structured_response_count = 0
        consecutive_provider_failures = 0
        max_provider_attempts = 3
        max_request_count = self._MAX_LLM_ATTEMPTS + 2
        while (
            request_count < max_request_count
            and structured_response_count < self._MAX_LLM_ATTEMPTS
        ):
            request_count += 1
            response_content: str | None = None
            try:
                response = await self._llm.complete(
                    current_messages,
                    reasoning_effort="high",
                )
                consecutive_provider_failures = 0
                response_content = response.content
                structured_response_count += 1
                usage = llm_call_usage(
                    response,
                    stage=usage_stage,
                    attempt=request_count,
                )
                usages.append(usage)
                self._log_token_usage(usage)
                return parser(response_content), request_count, tuple(usages)
            except LLMProviderError:
                last_failure = "provider request failed"
                consecutive_provider_failures += 1
                logger.warning(
                    "%s 的第 %d 次 Hy3 请求失败",
                    stage,
                    request_count,
                )
                if consecutive_provider_failures >= max_provider_attempts:
                    break
            except StructuredOutputError as exc:
                last_failure = str(exc)
                logger.warning(
                    "%s 的第 %d 次输出未通过校验：%s",
                    stage,
                    request_count,
                    exc,
                )
                if response_content is not None:
                    current_messages = (
                        *messages,
                        ChatMessage(role="assistant", content=response_content),
                        ChatMessage(
                            role="user",
                            content=(
                                f"上一次输出未通过结构校验：{exc} "
                                "请重新输出完整 JSON，不要解释。"
                            ),
                        ),
                    )
        raise ResearchGenerationError(
            f"Research {stage} failed after {request_count} LLM requests and "
            f"{structured_response_count} structured responses. "
            f"Last failure: {last_failure}",
            llm_request_count=request_count,
            llm_usages=usages,
        ) from None

    @staticmethod
    def _accumulated_generation_error(
        exc: ResearchGenerationError,
        *,
        llm_request_count: int,
        search_request_count: int,
        successful_search_count: int,
        llm_usages: Sequence[LLMCallUsage],
    ) -> ResearchGenerationError:
        """Attach completed earlier research stages to a terminal stage failure."""

        return ResearchGenerationError(
            str(exc),
            llm_request_count=llm_request_count + exc.llm_request_count,
            search_request_count=search_request_count + exc.search_request_count,
            successful_search_count=(
                successful_search_count + exc.successful_search_count
            ),
            llm_usages=(*llm_usages, *exc.llm_usages),
        )

    @staticmethod
    def _log_token_usage(usage: LLMCallUsage) -> None:
        logger.info(
            "Hy3 usage：stage=%s，input=%s，output=%s，total=%s",
            usage.stage,
            usage.input_tokens if usage.input_tokens is not None else "unknown",
            usage.output_tokens if usage.output_tokens is not None else "unknown",
            usage.total_tokens if usage.total_tokens is not None else "unknown",
        )

    @staticmethod
    def _validate_config(config: ResearchConfig) -> None:
        if not 1 <= config.initial_query_count <= 5:
            raise ValueError("initial_query_count must be between 1 and 5.")
        if not config.initial_query_count <= config.max_search_requests <= 10:
            raise ValueError("max_search_requests is outside the supported range.")
        if not 1 <= config.results_per_query <= 20:
            raise ValueError("results_per_query must be between 1 and 20.")
        if not 1 <= config.max_search_concurrency <= 10:
            raise ValueError("max_search_concurrency must be between 1 and 10.")
        if not 500 <= config.max_content_chars_per_result <= 20000:
            raise ValueError("max_content_chars_per_result is outside the supported range.")

    @staticmethod
    def _outcome(
        *,
        status: Literal["ready", "insufficient_evidence"],
        plan: QueryPlan,
        responses: Sequence[SearchResponse],
        evidence: Sequence[Evidence],
        claims: Sequence[Claim],
        errors: Sequence[str],
        llm_request_count: int,
        search_request_count: int,
        executed_queries: Sequence[PlannedQuery],
        llm_usages: Sequence[LLMCallUsage],
        title_chain: Sequence[TitleChainPart] = (),
    ) -> ResearchOutcome:
        return ResearchOutcome(
            status=status,
            query_plan=plan,
            search_responses=tuple(responses),
            evidence=tuple(evidence),
            claims=tuple(claims),
            errors=tuple(errors),
            query_plan_prompt_version=RESEARCH_QUERY_PLAN_PROMPT_VERSION,
            evidence_prompt_version=RESEARCH_EVIDENCE_PROMPT_VERSION,
            llm_request_count=llm_request_count,
            search_request_count=search_request_count,
            executed_queries=tuple(executed_queries),
            llm_usages=tuple(llm_usages),
            title_chain=tuple(title_chain),
        )
