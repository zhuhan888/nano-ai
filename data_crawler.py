from __future__ import annotations

import json
from pathlib import Path

from config import CRAWLED_DATA_PATH, get_config


def main() -> None:
    get_config()
    CRAWLED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "enabled": False,
        "message": "首版默认只使用合成数据训练。若后续需要联网采集，请在确认数据版权、清洗规则和目标站点后再扩展本脚本。",
        "output": str(CRAWLED_DATA_PATH),
        "samples": 0,
    }
    CRAWLED_DATA_PATH.write_text("", encoding="utf-8")
    print(json.dumps(placeholder, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
