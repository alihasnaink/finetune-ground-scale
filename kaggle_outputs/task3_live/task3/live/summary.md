### Live MATH strategies (50 questions, same 3,072-token output allowance; math-verify 0.9.0, last-literal-boxed-balanced-v2 (supplied-export rule))

| strategy | n_questions | accuracy_pct | final_no_answer_pct | responses_extracted_pct | responses_truncated_pct | mean_input_tokens | mean_output_tokens | output_allowance | mean_time_s | total_time_min | peak_mem_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 50 | 66.00 | 4.00 | 96.00 | 4.00 | 108.38 | 738.26 | 3072 | 34.67 | 28.89 | 6.09 |
| parallel | 50 | 62.00 | 10.00 | 86.67 | 14.00 | 325.14 | 1877.58 | 3072 | 86.25 | 71.88 | 5.94 |
| refine | 50 | 62.00 | 0.00 | 95.00 | 5.00 | 870.26 | 1141.72 | 3072 | 53.52 | 44.60 | 6.17 |

Shared initial response accuracy 62.0%; parallel oracle coverage (any of 3 correct) 68.0%; revised answer differs from the initial one in 10.0% of questions.

### Refinement transitions

| transition (initial -> revised) | questions |
|---|---|
| W->C | 0 |
| C->W | 0 |
| C->C | 31 |
| W->W | 19 |

### Paired accuracy differences (1,000 question bootstrap resamples)

| comparison | accuracy diff (pp) | 95% CI |
|---|---|---|
| parallel_minus_single | -4.00 | [-16.0, 8.0] |
| refine_minus_single | -4.00 | [-14.0, 8.0] |
| refine_minus_parallel | 0.00 | [-6.0, 6.0] |
| refine_minus_initial | 0.00 | [0.0, 0.0] |
| parallel_minus_initial | 0.00 | [-6.0, 6.0] |
