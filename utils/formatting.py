from __future__ import annotations
import textwrap

def plan_json_to_markdown(plan: dict) -> str:
    lines = []
    title = plan.get("title", "愛媛 旅程プラン")
    summary = plan.get("summary", "")
    lines.append(f"# {title}")
    if summary:
        lines.append(summary)
        lines.append("")
    for day in plan.get("days", []):
        d = day.get("day")
        theme = day.get("theme", "")
        area = day.get("area", "")
        lines.append(f"## Day {d}: {theme} ({area})")
        for s in day.get("schedule", []):
            time = s.get("time", "")
            spot = s.get("spot", "")
            act = s.get("activity", "")
            tip = s.get("tip", "")
            url = s.get("url", "")
            addr = s.get("address", "")
            bullet = f"- **{time}** {spot} — {act}"
            if addr:
                bullet += f"｜{addr}"
            if url:
                bullet += f" ｜[公式情報]({url})"
            if tip:
                bullet += f"n  t> メモ: {tip}"
            lines.append(bullet)
        # 出典URL
        srcs = day.get("source_urls", [])
        if srcs:
            lines.append("n**根拠URL**:")
            for u in srcs:
                lines.append(f"- {u}")
        lines.append("")

    if plan.get("sources"):
        lines.append("---n## 参考ソース")
        for s in plan["sources"]:
            lines.append(f"- [{s['title']}]({s['url']}) — {s.get('site','')}")
    return "n".join(lines)