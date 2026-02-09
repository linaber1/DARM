from typing import List

def stitch(plan: str, refined: str, extras: List[str], headers) -> str:
    parts = []
    if extras:
        parts += [headers.concat_header_extra, "\n".join(e.strip() for e in extras if e.strip())]
    if plan:
        parts += [headers.concat_header_plan, plan.strip()]
    if refined:
        parts += [headers.concat_header_refined, refined.strip()]
    
    return "\n\n".join(parts).strip()
