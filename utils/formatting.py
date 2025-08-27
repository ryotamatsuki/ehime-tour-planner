from __future__ import annotations
import textwrap

def plan_json_to_markdown(plan: dict) -> str:
    lines = []
    title = plan.get("title", "愛媛 旅程プラン")
    summary = plan.get("summary", "")

    # Use smaller headings for a cleaner look
    lines.append(f"### {title}")
    if summary:
        lines.append(f"*{summary}*")
    lines.append("")

    for day in plan.get("days", []):
        d = day.get("day")
        theme = day.get("theme", "")
        area = day.get("area", "")
        
        # Use a smaller heading for the day's theme
        lines.append(f"#### Day {d}: {theme} ({area})")
        
        for s in day.get("schedule", []):
            time = s.get("time", "")
            spot = s.get("spot", "")
            act = s.get("activity", "")
            tip = s.get("tip", "")
            url = s.get("url", "")
            addr = s.get("address", "")
            
            # De-emphasize time, emphasize the spot, use a colon
            line = f"- {time} **{spot}**: {act}"
            
            details = []
            if addr:
                details.append(f"住所: {addr}")
            if url:
                details.append(f"[公式情報]({url})")

            # Combine address and URL if both exist
            if details:
                line += f" ({'｜'.join(details)})"
            
            lines.append(line)

            # Indent tip and make it italic for a softer look
            if tip:
                lines.append(f"  - *メモ: {tip}*")
        
        lines.append("") # Add space after each day's schedule
        
        srcs = day.get("source_urls", [])
        if srcs:
            lines.append("**根拠URL**:")
            for u in srcs:
                lines.append(f"- {u}")
        lines.append("")

    if plan.get("sources"):
        lines.append("---")
        lines.append("## 参考ソース")
        for s in plan["sources"]:
            lines.append(f"- [{s['title']}]({s['url']}) — {s.get('site','')}")
            
    return "\n".join(lines)