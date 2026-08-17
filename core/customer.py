import re

FIELD_RULES = {
    "name": ["姓名", "名字"],
    "age": ["年龄", "年齡", "年纪", "年紀"],
    "job": ["职业", "職業"],
    "income": ["收入"],
    "work_year": ["工作年限", "工作年數", "工作年数"],
    "software": ["引流软件", "引流軟件", "引流平台"],
    "receiver": ["接粉人员", "接粉人員", "接粉人"],
}


def _clean_value(v):
    v = re.sub(r"^[\s:：=\-]+", "", v or "").strip()
    return re.sub(r"[\s,，;；|/]+$", "", v).strip()


def _label_pattern():
    pairs = []
    for key, aliases in FIELD_RULES.items():
        for alias in aliases:
            pairs.append((alias, key))
    # Long labels first so 工作年限 is not partially confused by shorter text.
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    labels = "|".join(re.escape(x[0]) for x in pairs)
    return pairs, re.compile(rf"(?<![\w\u4e00-\u9fff])(?P<label>{labels})\s*[:：=]?\s*", re.I)


_LABEL_PAIRS, _FIELD_RE = _label_pattern()
_ALIAS_TO_KEY = {a.lower(): k for a, k in _LABEL_PAIRS}


def parse_customer_info(text):
    """Parse the user's customer-card format from Telegram text/caption/OCR.

    Supports both the usual one-field-per-line format and image captions where
    several fields are written on one line, e.g.:
    姓名：bibi 年龄：39 职业：文职 工作年限：20年 引流软件：小软件+1
    """
    data = {k: "" for k in FIELD_RULES}
    labels_found = set()
    if not text:
        data["_labels_found"] = labels_found
        return data

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_FIELD_RE.finditer(normalized))
    if not matches:
        data["_labels_found"] = labels_found
        return data

    for idx, m in enumerate(matches):
        label = m.group("label")
        key = _ALIAS_TO_KEY.get(label.lower())
        if not key:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        value = normalized[start:end]
        # Strip line breaks/separators left between adjacent fields.
        value = re.sub(r"^[\s,，;；|/]+", "", value)
        value = re.sub(r"[\s,，;；|/]+$", "", value)
        data[key] = _clean_value(value)
        labels_found.add(key)

    data["_labels_found"] = labels_found
    return data


def is_customer_record(data):
    labels = set(data.get("_labels_found") or [])
    # 正式资料仍以姓名为核心，至少再出现两个字段标签，避免普通聊天误入库。
    return "name" in labels and len(labels) >= 3


def public_customer_data(data):
    return {k: data.get(k, "") for k in FIELD_RULES}
