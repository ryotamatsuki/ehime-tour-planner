from __future__ import annotations


def plan_json_to_markdown(plan: dict) -> str:
    lines = []
    title = plan.get("title", "愛媛 旅程プラン")
    summary = plan.get("summary", "")

    lines.append(f"### {title}")
    if summary:
        lines.append(f"*{summary}*")
    lines.append("")

    for day in plan.get("days", []):
        d = day.get("day")
        theme = day.get("theme", "")
        area = str(day.get("area", "") or "").strip()
        area_suffix = f" ({area})" if area else ""
        lines.append(f"#### Day {d}: {theme}{area_suffix}")

        for schedule in day.get("schedule", []):
            time = schedule.get("time", "")
            spot = schedule.get("spot", "")
            activity = schedule.get("activity", "")
            tip = schedule.get("tip", "")
            url = schedule.get("url", "")
            address = schedule.get("address", "")

            line = f"- {time} **{spot}**: {activity}"
            details = []
            if address:
                details.append(f"住所: {address}")
            if url:
                details.append(f"[公式情報]({url})")
            if details:
                line += f" ({'｜'.join(details)})"
            lines.append(line)

            if tip:
                lines.append(f"  - *メモ: {tip}*")

        lines.append("")
        source_urls = day.get("source_urls", [])
        if source_urls:
            lines.append("**根拠URL**:")
            for url in source_urls:
                lines.append(f"- {url}")
        lines.append("")

    if plan.get("sources"):
        lines.append("---")
        lines.append("## 参考ソース")
        for source in plan["sources"]:
            lines.append(
                f"- [{source['title']}]({source['url']}) — {source.get('site', '')}"
            )

    return "\n".join(lines)
