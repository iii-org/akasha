from __future__ import annotations

import sys


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "\u670b\u53cb"
    print(f"您好 {name}！這是來自 Akasha Skill script 的問候。")


if __name__ == "__main__":
    main()
