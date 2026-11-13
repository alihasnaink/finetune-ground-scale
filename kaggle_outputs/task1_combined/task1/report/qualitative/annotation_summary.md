### Qualitative annotation summary (42 responses: 2 questions × 3 formats × 7 conditions)

Per condition, over 6 responses (2 questions × 3 formats). format_ok counts the 4 JSON / template responses (plain_ok: plain answers judged natural / unformatted where rated); coherence 2 = coherent and on topic; repetition / math framing: 1 = some / mild, 2 = loop / strong; truncated = hit the 512-token cap.

| condition | annotated | truncated | format_ok json+template | plain_ok | coherence mean (0-2) | repetition some/loop | math framing mild/strong | inappropriate |
|---|---|---|---|---|---|---|---|---|
| baseline | 6/6 | 0 | 4/4 | 2/2 | 1.83 | 0/0 | 0/0 | 0 |
| full_ft_mid | 6/6 | 2 | 3/4 | 2/2 | 1.33 | 1/2 | 0/0 | 0 |
| full_ft_final | 6/6 | 2 | 3/4 (+1 partial) | 0/2 (+2 partial) | 1.33 | 0/2 | 3/1 | 0 |
| l2sp_final | 6/6 | 0 | 2/4 (+2 partial) | 2/2 | 1.67 | 0/0 | 3/0 | 0 |
| lora4_final | 6/6 | 0 | 3/4 | 2/2 | 1.67 | 0/0 | 0/0 | 0 |
| lora16_final | 6/6 | 1 | 2/4 (+1 partial) | 2/2 | 1.83 | 0/1 | 1/0 | 0 |
| oplora_final | 6/6 | 0 | 3/4 | 2/2 | 1.83 | 1/0 | 2/0 | 0 |


### Format compliance codes per question (y / p / n), first then last question

| condition | json | template |
|---|---|---|
| baseline | yy | yy |
| full_ft_mid | yn | yy |
| full_ft_final | yp | yy |
| l2sp_final | yp | yp |
| lora4_final | yn | yy |
| lora16_final | yn | yp |
| oplora_final | yn | yy |


### Annotated excerpts

