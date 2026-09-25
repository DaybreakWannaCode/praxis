"""Fixed answer-label likelihood surrogate. Never generates reasoning responses.

This objective is NOT expected correctness under the original Praxis policy.
Labels are normalized short token-sequence prefix likelihoods (no EOS).
"""
import torch
from task_0.src.rewards import option_labels

SYSTEM_PROMPT = 'Answer the visual multiple-choice question. Reply with only the option label, without explanation.'


def messages(item):
    # The target is intentionally absent from prompt construction.
    return [dict(role='system', content=[dict(type='text', text=SYSTEM_PROMPT)]),
            dict(role='user', content=[dict(type='image'), dict(type='text',
                 text=item.question + '\n' + '\n'.join(item.action_list))])]


def label_tokens(tokenizer, labels):
    rows = [tokenizer.encode(x, add_special_tokens=False) for x in labels]
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not x for x in rows):
        raise ValueError('Invalid labels')
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i != j and a == b[:len(a)]:
                raise ValueError('Tokenized labels must be distinct and prefix-free')
    return rows


def normalized_target(log_weights, target):
    if log_weights.ndim != 1 or not 0 <= target < log_weights.numel():
        raise ValueError('Invalid target or choice scores')
    if not torch.isfinite(log_weights).all():
        raise ValueError('Nonfinite label scores')
    # Float64 reduction preserves small differences; model itself is FP32.
    scores = log_weights.double()
    return scores[target] - torch.logsumexp(scores, dim=0)


def prepare(backend, item):
    proc = backend.extractor.proc
    rendered = proc.apply_chat_template(messages(item), tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[rendered], images=[item.image()], return_tensors='pt')
    inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
    labels = option_labels(item.action_list)
    tokens = label_tokens(proc.tokenizer, labels)
    return inputs, labels, tokens


def log_weights(backend, inputs, tokens):
    model = backend.model
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None
    if all(len(t) == 1 for t in tokens):
        kwargs = dict(inputs, use_cache=False)
        if backend.extractor._ltk_kw:
            kwargs[backend.extractor._ltk_kw] = 1
        logits = model(**kwargs).logits[0, -1]
        # Full-vocabulary normalizer cancels when all alternatives are one token.
        return logits[[t[0] for t in tokens]].double()
    prompt = inputs['input_ids'][0]
    vis = {k: v for k, v in inputs.items() if k in ('pixel_values', 'image_grid_thw')}
    values = []
    for ids in tokens:
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None
        seq = torch.cat((prompt, torch.tensor(ids, device=prompt.device)))
        values.append(backend.extractor.sequence_logprob(seq, len(prompt), vis)[0].double())
    return torch.stack(values)


def objective(backend, item):
    inputs, labels, tokens = prepare(backend, item)
    return normalized_target(log_weights(backend, inputs, tokens), labels.index(item.answer)), tokens
