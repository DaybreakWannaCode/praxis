"""Development-audited repair of malformed answer tags; v2 stays immutable."""
import re

from .answer_parsing import parse_choice as parse_v2
from task_0.src.rewards import option_labels

_OPEN = re.compile(r'<([A-Za-z]?)answer>',re.I)
_CLOSE = re.compile(r'</answer>',re.I)
_HEAD = re.compile(r'^[\s*_`\[(]*([A-Za-z])(?=$|[\s).:,;\]>*_`])')
_MULTI = re.compile(r'^[\s*_`\]).,;>]*(?:or|and|/)\s*\(?([A-Za-z])\b',re.I)


def parse_choice(completion,action_list,*,profile='explicit_final_v3'):
    if profile!='explicit_final_v3':raise ValueError('Unknown v3 parser profile')
    text=completion or ''
    opening=_OPEN.search(text)
    if opening is None:return parse_v2(text,action_list,profile='explicit_final_v2')
    tail=text[opening.end():]
    closing=_CLOSE.search(tail)
    # A missing closing tag must never allow a later reasoning option to override
    # the answer at the start of this span (e.g. <answer>B> ... Option A ...).
    span=tail[:closing.start()] if closing else tail.split('<',1)[0]
    head=_HEAD.match(span)
    labels=set(option_labels(action_list))
    if head:
        choice=head.group(1).upper()
        if choice not in labels:return None
        stray=opening.group(1).upper()
        # Tolerate one duplicated label in the opening tag only when it agrees
        # with the explicit answer: <Banswer>B</answer>. No arbitrary tag repair.
        if stray and stray!=choice:return None
        alternative=_MULTI.match(span[head.end():])
        if alternative and alternative.group(1).upper() in labels:return None
        return choice
    if not closing or opening.group(1):return None
    # Preserve v2's closed, prose-only answer semantics, scoped to the answer.
    return parse_v2('<answer>'+span+'</answer>',action_list,profile='explicit_final_v2')


def score_completion(completion,gold,action_list,*,profile='explicit_final_v3'):
    choice=parse_choice(completion,action_list,profile=profile)
    return float(choice==str(gold).strip().upper()) if choice is not None else 0.,choice is not None
