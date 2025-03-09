import spacy
from spacy.tokens import Span, Doc
from spacy.language import Language
from spacy.pipeline import EntityRuler
from dateparser import parse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from typing import List, Tuple, Optional

# ======================
# 配置区（可扩展）
# ======================
TIME_PATTERNS = [
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["between", "from"]}},
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{4}$"}},
            {"LOWER": {"IN": ["to", "and"]}},
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{4}$"}}
        ],
        "type": "absolute",
        "handler": "handle_cross_year_range"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["last", "past", "previous"]}},
            {"LOWER": {"IN": ["three", "two", "one", "four", "five", "six", "seven", "eight", "nine", "ten"]}},
            {"LOWER": {"REGEX": r"^months?$"}}
        ],
        "type": "relative",
        "handler": "handle_relative_months_words"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["next", "coming"]}},
            {"LOWER": {"IN": ["three", "two", "one", "four", "five", "six", "seven", "eight", "nine", "ten"]}},
            {"LOWER": {"REGEX": r"^months?$"}}
        ],
        "type": "relative",
        "handler": "handle_relative_months_words"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{4}$"}},
            {"LOWER": {"IN": ["to", "and"]}},
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{4}$"}}
        ],
        "type": "absolute",
        "handler": "handle_range_month_year"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{4}$"}}
        ],
        "type": "absolute",
        "handler": "handle_month_year"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["previous", "last"]}},
            {"LOWER": "quarter"}
        ],
        "type": "relative",
        "handler": "handle_relative_quarter"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["next", "coming"]}},
            {"LOWER": "quarter"}
        ],
        "type": "relative",
        "handler": "handle_relative_quarter"
    },

    # 优化月份年份模式
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},  # 完整列出12个月份
            {"TEXT": {"REGEX": r"^\d{1,2}(?:st|nd|rd|th)?,?"}},
            {"TEXT": {"REGEX": r"^\d{4}$"}, "OP": "?"}
        ],
        "type": "absolute",
        "handler": "handle_month_date"
    },
    # 绝对时间模式（修正后的季度匹配）
    {
        "label": "TIME",
        "pattern": [
            {"TEXT": {"REGEX": r"(?i)^Q[1-4]$"}},  # 修饰符 (?i) 必须在最前
            {"TEXT": {"REGEX": r"^\d{4}$"}}
        ],
        "type": "absolute",
        "handler": "handle_quarter"
    },
    # 月份日期格式（修正大小写问题）
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"TEXT": {"REGEX": r"^\d{1,2}(?:st|nd|rd|th)?,?"}},
            {"TEXT": {"REGEX": r"^\d{4}$"}, "OP": "?"}
        ],
        "type": "absolute",
        "handler": "handle_month_date"
    },

    # 相对时间模式（保持原样）
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["last", "past", "previous"]}},
            {"LOWER": {"REGEX": r"^\d+$"}},
            {"LOWER": {"REGEX": r"^months?$"}}
        ],
        "type": "relative",
        "handler": "handle_relative_months"
    },
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"IN": ["next", "coming", "following"]}},
            {"LOWER": {"REGEX": r"^\d+$"}},
            {"LOWER": {"REGEX": r"^months?$"}}
        ],
        "type": "relative",
        "handler": "handle_relative_months"
    },

    # 持续时间模式（保持原样）
    {
        "label": "TIME",
        "pattern": [
            {"LOWER": {"REGEX": r"^\d+$"}},
            {"LOWER": {"REGEX": r"^months?$"}}
        ],
        "type": "duration",
        "handler": "handle_duration"
    }
]


# ======================
# 核心处理类
# ======================
class TimeParser:
    def __init__(self):
        self.time_anchors = []
        self.current_date = datetime.now()
        self.doc_char_offset = 0  # 新增文档级字符偏移量

    def handle_cross_year_range(self, ent: Span) -> Tuple[datetime, datetime]:
        month1 = ent[1].text.lower()
        year1 = int(ent[2].text)
        month2 = ent[4].text.lower()
        year2 = int(ent[5].text)

        month_map = {...}  # 同前

        start_date = datetime(year1, month_map[month1], 1)
        end_date = datetime(year2, month_map[month2], 1) + relativedelta(day=31)

        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })

        if end_date < start_date:
            start_date, end_date = end_date, start_date
            # 记录调试信息
        print(f"跨年度处理：{ent.text} -> {start_date.date()}~{end_date.date()}")
        return (start_date, end_date)

    def handle_relative_quarter(self, ent: Span, base_date: datetime) -> Tuple[datetime, datetime]:
        direction = -1 if ent[0].lower_ in ["previous", "last"] else 1
        base_quarter = (base_date.month - 1) // 3 + 1
        target_quarter = base_quarter + direction

        year = base_date.year
        if target_quarter < 1:
            year -= 1
            target_quarter = 4
        elif target_quarter > 4:
            year += 1
            target_quarter = 1

        start_month = 3 * (target_quarter - 1) + 1
        end_month = 3 * target_quarter
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month, 1) + relativedelta(day=31)

        # 记录时间基准
        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })

        return (start_date, end_date)

    def handle_relative_months(self, ent: Span, base_date: datetime) -> Tuple[datetime, datetime]:
        direction = -1 if ent[0].lower_ in ["last", "past", "previous"] else 1
        num_months = int(ent[1].text)

        # 根据方向计算基准点
        if direction == -1:
            end_date = base_date.replace(day=1) - timedelta(days=1)
            start_date = end_date - relativedelta(months=num_months - 1)
            start_date = start_date.replace(day=1)
        else:
            start_date = base_date.replace(day=1) + relativedelta(months=1)
            end_date = start_date + relativedelta(months=num_months) - timedelta(days=1)


        # 记录时间基准
        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })

        # 记录调试信息
        print(f"相对月份计算：基准={base_date.date()} 方向={direction} 数量={num_months}")

        return (start_date, end_date)

    def handle_relative_months_words(self, ent: Span, base_date: datetime) -> Tuple[datetime, datetime]:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        direction = -1 if ent[0].lower_ in ["last", "past", "previous"] else 1
        num_months = word_to_num.get(ent[1].text.lower(), 0)

        if num_months == 0:
            return (None, None)

        # 基于上下文基准日期计算
        calc_date = base_date + relativedelta(months=direction * num_months)

        if direction == -1:
            start_date = calc_date.replace(day=1)
            end_date = base_date.replace(day=1) - timedelta(days=1)
        else:
            start_date = base_date + timedelta(days=1)
            end_date = (calc_date + relativedelta(day=31))

        # 确保日期顺序正确
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        # 记录新的时间锚点
        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })
        return (start_date, end_date)

    def parse_absolute(self, ent: Span) -> Optional[Tuple[datetime, datetime]]:
        handler_name = ent._.time_config.get("handler")
        handler = getattr(self, handler_name, None)
        if handler:
            return handler(ent)
        return None

    def parse_relative(self, ent: Span) -> Optional[Tuple[datetime, datetime]]:
        # 自动查找最近的时间基准
        base_date = self._find_context_base(ent)
        handler_name = ent._.time_config.get("handler")
        handler = getattr(self, handler_name, None)
        if handler:
            return handler(ent, base_date)
        return None

    def _find_context_base(self, ent: Span) -> datetime:
        """基于字符位置的精准查找"""
        current_char = ent.start_char
        # 筛选有效锚点（在当前实体左侧）
        valid_anchors = [a for a in self.time_anchors if a["char_end"] <= current_char]

        if not valid_anchors:
            return self.current_date

        # 选择最近的锚点（最大char_end值）
        nearest_anchor = max(valid_anchors, key=lambda x: x["char_end"])
        print(f"DEBUG: 实体[{ent.text}]@{ent.start_char} 使用锚点@{nearest_anchor['char_end']}")
        return nearest_anchor["end_date"]

    # ======================
    # 具体处理函数
    # ======================
    def handle_quarter(self, ent: Span) -> Tuple[datetime, datetime]:
        text = ent.text.lower().replace(" ", "")
        match = re.match(r'q?(\d)(\d{4})', text)
        if not match:
            return None

        q_num = int(match.group(1))
        year = int(match.group(2))

        if not (1 <= q_num <= 4):
            return None

        start_month = 3 * (q_num - 1) + 1
        end_month = 3 * q_num

        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month, 1) + relativedelta(day=31)

        # 记录时间基准
        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })

        return (start_date, end_date)

    def handle_relative_months(self, ent: Span, base_date: datetime) -> Tuple[datetime, datetime]:
        direction = -1 if ent[0].lower_ in ["last", "past", "previous"] else 1
        num_months = int(ent[1].text)

        start_date = base_date + relativedelta(months=direction * num_months)
        end_date = base_date

        # 自动调整日期边界
        if direction == -1:
            return (
                start_date.replace(day=1),
                end_date.replace(day=1) - timedelta(days=1)
            )
        else:
            return (
                end_date.replace(day=1) + timedelta(days=1),
                start_date + relativedelta(day=31)
            )

    def handle_month_year(self, ent: Span) -> Tuple[datetime, datetime]:
        """处理纯月份+年份格式（如 March 2024）"""
        month_str = ent[0].text.lower()
        year = int(ent[1].text)

        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        start_date = datetime(year, month_map[month_str], 1)
        end_date = start_date + relativedelta(months=1, days=-1)

        # 记录时间基准
        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })
        return (start_date, end_date)

    def handle_duration(self, ent: Span) -> Tuple[datetime, datetime]:
        """处理持续时间（如 6 months）"""
        duration = int(ent[0].text)
        unit = ent[1].text.lower()

        # 自动查找最近的基准时间
        base_date = self._find_context_base(ent)

        if unit.startswith("month"):
            start_date = base_date
            end_date = base_date + relativedelta(months=duration)
            # 调整到月末最后一天
            end_date = end_date.replace(day=1) + relativedelta(days=-1)
            return (start_date, end_date)
        return (None, None)

    def handle_month_date(self, ent: Span) -> Tuple[datetime, datetime]:
        """处理带日期的月份格式（如 March 15th, 2024）"""
        # 实现逻辑（示例）
        month_str = ent[0].text.lower()
        day = int(re.search(r'\d+', ent[1].text).group())
        year = int(ent[2].text) if len(ent) > 2 else self.current_date.year

        month_map = {...}  # 同handle_month_year
        start_date = datetime(year, month_map[month_str], day)
        end_date = start_date  # 单日范围
        return (start_date, end_date)

    def handle_range_month_year(self, ent: Span) -> Tuple[datetime, datetime]:
        month1 = ent[0].text.lower()
        year1 = int(ent[1].text)
        month2 = ent[3].text.lower()
        year2 = int(ent[4].text)

        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        start_date = datetime(year1, month_map[month1], 1)
        end_date = datetime(year2, month_map[month2], 1) + relativedelta(months=1, days=-1)

        self.time_anchors.append({
            "char_start": ent.start_char,  # 使用字符级位置
            "char_end": ent.end_char,
            "start_date": start_date,
            "end_date": end_date
        })

        return (start_date, end_date)


# ======================
# SpaCy 组件
# ======================
@Language.factory("time_parser")
def create_time_parser(nlp: Language, name: str):
    return TimeParserComponent()


class TimeParserComponent:
    def __init__(self):
        self.parser = None  # 不再共享实例

    def __call__(self, doc: Doc) -> Doc:
        # 建立字符位置映射表
        char_pos_map = {}
        for token in doc:
            char_pos_map[token.idx] = token.i
        doc.user_data["char_pos_map"] = char_pos_map

        self.parser = TimeParser()  # 每个文档创建新实例
        # 第一次处理：识别所有时间表达式
        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns([p for p in TIME_PATTERNS if "pattern" in p])
        doc = ruler(doc)

        # 标记时间类型和处理程序
        for ent in doc.ents:
            for pattern in TIME_PATTERNS:
                if self._match_pattern(ent, pattern["pattern"]):
                    ent._.set("time_type", pattern["type"])
                    ent._.set("time_config", pattern)
                    break

        # 第二次处理：解析时间范围
        for ent in doc.ents:
            if ent._.time_type == "absolute":
                time_range = self.parser.parse_absolute(ent)
            elif ent._.time_type == "relative":
                time_range = self.parser.parse_relative(ent)
            else:
                continue

            if time_range:
                ent._.set("time_range", time_range)
                ent._.set("display",
                          f"{time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}")

        print(f"\n处理文档：{doc.text}")
        print("发现的时间锚点轨迹：")
        for anchor in self.parser.time_anchors:
            print(f"位置{anchor['position']}: {anchor['start_date'].date()} 至 {anchor['end_date'].date()}")

        return doc

    def _match_pattern(self, ent: Span, pattern: List[dict]) -> bool:
        """验证实体是否匹配指定模式"""
        if len(ent) != len(pattern):
            return False
        for token, cond in zip(ent, pattern):
            if not self._check_token(token, cond):
                return False
        return True

    def _check_token(self, token, cond: dict) -> bool:
        """检查单个token是否符合条件"""
        for attr, rule in cond.items():
            if attr == "LOWER":
                if isinstance(rule, dict):
                    if "IN" in rule:
                        if token.lower_ not in rule.get("IN", []):
                            return False
                    elif "REGEX" in rule:
                        if not re.match(rule["REGEX"], token.lower_):
                            return False
                elif isinstance(rule, str):
                    if token.lower_ != rule:
                        return False
                else:
                    return False
            elif attr == "TEXT":
                if isinstance(rule, dict):
                    if "REGEX" in rule and not re.match(rule["REGEX"], token.text):
                        return False
                    if "IN" in rule and token.text not in rule["IN"]:
                        return False
                elif isinstance(rule, str):
                    if token.text != rule:
                        return False
                else:
                    return False
        return True


# ======================
# 初始化
# ======================
nlp = spacy.load("en_core_web_sm")
Span.set_extension("time_type", default=None)
Span.set_extension("time_config", default=None)
Span.set_extension("time_range", default=None)
Span.set_extension("display", default=None)

nlp.add_pipe("time_parser", last=True)

# ======================
# 使用示例
# ======================
texts = [
    "Q2 2023 report shows growth from last three months compared to Q1 2023",
    "The project starting in March 2024 will take 6 months",
    "Previous quarter results were better than next 2 months forecast",
    "Between April 2023 and June 2023, we observed significant changes"
]

for text in texts:
    doc = nlp(text)
    print(f"\n文本：{text}")
    for ent in doc.ents:
        if ent.label_ == "TIME" and ent._.time_range:
            start, end = ent._.time_range
            print(f"- {ent.text} ({ent._.time_type})")
            print(f"  ↳ {ent._.display}")
            print(f"  ↳ 开始：{start.date()}，结束：{end.date()}")