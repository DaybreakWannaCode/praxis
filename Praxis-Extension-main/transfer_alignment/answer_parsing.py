"""Versioned correction for explicit final choices in untagged completions."""
import re
from task_0.src.rewards import option_labels, parse_choice as legacy_parse

# A terminal declaration is more authoritative than options discussed in reasoning.
# Require a line boundary; ordinary prose such as 'the answer could be A' is not one.
_DECLARATION = re.compile(
    r'^[ \t*#_-]*(?:final\s+(?:answer|option|choice)|answer)\s*(?:is\s*)?[:=]?\s*'
    r'[*_`\[(]*([A-Za-z])\b(.*)$', re.IGNORECASE | re.MULTILINE)
_TAG = re.compile(r'<answer>', re.IGNORECASE)


def parse_choice(completion, action_list, *, profile='explicit_final_v2'):
    if profile=='legacy':return legacy_parse(completion,action_list)
    if profile!='explicit_final_v2':raise ValueError('Unknown answer parser profile')
    text=completion or ''
    # Preserve existing tagged-answer semantics; this version corrects the untagged path.
    if _TAG.search(text):return legacy_parse(text,action_list)
    declarations=list(_DECLARATION.finditer(text))
    if not declarations:return legacy_parse(text,action_list)
    last=declarations[-1];choice=last.group(1).upper();suffix=last.group(2)
    labels=set(option_labels(action_list))
    if choice not in labels:return None
    # Refuse an explicit multi-choice answer, rather than falling back into the reasoning.
    alternative=re.match(r'^[\s*_`\]).,]*(?:or|and|/)\s*\(?([A-Za-z])\b',suffix,re.I)
    if alternative and alternative.group(1).upper() in labels:return None
    return choice


def score_completion(completion,gold,action_list,*,profile='explicit_final_v2'):
    choice=parse_choice(completion,action_list,profile=profile)
    return float(choice==str(gold).strip().upper()) if choice is not None else 0., choice is not None
