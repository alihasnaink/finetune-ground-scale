# Task 3: representative cases

Selection rules are fixed in advance and stated per section; they use the saved orders and the held-out split order. Gold answers are used only to pick illustrations, never by any method.

## MATH (Qwen2.5-3B-Instruct)

### Selection disagreement (rule: first held-out question in split order where, at N = 32 and order 0, self-consistency and SelfCert-RWC differ in correctness; one per direction)

**SelfCert-RWC right, SC wrong**: question `test:2744`, gold `2`, 15/32 correct. SC picks `0`; SelfCert-RWC picks `2`.

| answer | correct | votes | RWC weight | mean K |
|---|---|---|---|---|
| 0 | False | 16 | 5.516 | 30.067 |
| 2 | True | 15 | 10.258 | 30.867 |
| \frac{7 + \sqrt{29}}{2} | False | 1 | 0.226 | 29.914 |

**SC right, SelfCert-RWC wrong**: question `test:2198`, gold `11`, 5/32 correct. SC picks `11`; SelfCert-RWC picks `14`.

| answer | correct | votes | RWC weight | mean K |
|---|---|---|---|---|
| 11 | True | 5 | 0.806 | 26.456 |
| 14 | False | 4 | 3.161 | 30.043 |
| 10^{200} - 7 \cdot 10^{100} +  | False | 3 | 2.258 | 29.208 |
| 6 | False | 2 | 0.871 | 27.573 |
| 10^{200} - 7 \cdot 10^{100} +  | False | 2 | 1.581 | 29.305 |

### Premature stopping (rule: first held-out question in split order, lowest order index, where the frozen agreement policy (tau = 0.625) stops early with a wrong answer that fixed-32 self-consistency gets right)

Question `test:2460` (order 7), gold `154`, 11/32 correct in the full pool. Stopped at n = 4 with answer `58`.

Answer groups among the first 4:

| answer | correct | votes | RWC weight | mean K |
|---|---|---|---|---|
| 58 | False | 3 | 1.667 | 27.659 |
| 154 | True | 1 | 0.333 | 26.864 |

All 32 candidates:

| answer | correct | votes | RWC weight | mean K |
|---|---|---|---|---|
| 154 | True | 11 | 6.806 | 27.384 |
| 58 | False | 11 | 5.806 | 27.270 |
| NO_ANSWER | False | 3 | 0.645 | 26.166 |
| 90 | False | 1 | 0.710 | 27.461 |
| 98 | False | 1 | 0.097 | 26.030 |

End of a leading-group rollout (24): "..., then \( 2^{k+1} = 2^6 = 64 \), and: \[ N = 64 - 19 = 45. \] So, the two smallest possible values of \( N \) are 13 and 45. The sum of these two values is: \[ 13 + 45 = 58. \] Thus, the sum of the two smallest possible values of \( N \) is \(\boxed{58}\)."

Totals (held-out, agreement policy): 60 of 5884 early stops are wrong-but-recoverable by fixed-32 SC.

## MathVista (Qwen2.5-VL-3B-Instruct)

### Selection disagreement (rule: first held-out question in split order where, at N = 32 and order 0, self-consistency and VisConf-RWC differ in correctness; one per direction)

**VisConf-RWC right, SC wrong**: question `643`, gold `A`, 10/32 correct. SC picks `C`; VisConf-RWC picks `A`.

| answer | correct | votes | RWC weight | mean Z |
|---|---|---|---|---|
| C | False | 11 | 4.129 | 0.005 |
| A | True | 10 | 5.452 | 0.005 |
| NO_ANSWER | False | 8 | 5.161 | 0.006 |
| D | False | 3 | 1.258 | 0.005 |

**SC right, VisConf-RWC wrong**: question `552`, gold `4`, 14/32 correct. SC picks `4`; VisConf-RWC picks `2`.

| answer | correct | votes | RWC weight | mean Z |
|---|---|---|---|---|
| 4 | True | 14 | 4.774 | 0.059 |
| 2 | False | 9 | 7.419 | 0.068 |
| 3 | False | 9 | 3.806 | 0.061 |

### Premature stopping (rule: first held-out question in split order, lowest order index, where the frozen agreement policy (tau = 0.625) stops early with a wrong answer that fixed-32 self-consistency gets right)

Question `847` (order 6), gold `A`, 21/32 correct in the full pool. Stopped at n = 4 with answer `C`.

Answer groups among the first 4:

| answer | correct | votes | RWC weight | mean Z |
|---|---|---|---|---|
| C | False | 3 | 1.667 | 0.033 |
| B | False | 1 | 0.333 | 0.030 |

All 32 candidates:

| answer | correct | votes | RWC weight | mean Z |
|---|---|---|---|---|
| A | True | 21 | 10.065 | 0.029 |
| C | False | 6 | 3.484 | 0.030 |
| B | False | 5 | 2.452 | 0.029 |

End of a leading-group rollout (14): "...\(\log_2 x\) but slower than \(\log_{10} x\). - \(\log_{10} x\) has the largest base among the three, making it the fastest-growing function. Therefore, the function that grows the fastest as \(x\) increases is \(\log_{10} x\). \(\boxed{\text{C. blue}}\)"

Totals (held-out, agreement policy): 133 of 5463 early stops are wrong-but-recoverable by fixed-32 SC.

## Step 4: sequential refinement (MATH live subset)

Rule: every question whose revised answer differs from the shared initial answer, then the first question (live index order) of each transition class that occurs.

| qid | gold | initial | initial_tokens | initial_truncated | revised | revision_tokens | transition | revision_opening |
|---|---|---|---|---|---|---|---|---|
| test:3285 | 0 | NO_ANSWER | 1024 | True | 302562 | 1149 | W->W | Let's carefully re-evaluate the steps to ensure accuracy. ### Step 1: Sum the numbers as if they were in base |
| test:2177 | 18 | NO_ANSWER | 1024 | True | \frac{49}{2} | 977 | W->W | Let's carefully review the solution and identify any potential errors. We start with the given equations: \[ a |
| test:2358 | 2 \sqrt{5} | NO_ANSWER | 1024 | True | 2\sqrt{3} | 921 | W->W | To solve the problem correctly, we need to carefully re-evaluate the steps and ensure all calculations are acc |
| test:4678 | 6 | NO_ANSWER | 1024 | True | 4 | 799 | W->W | Let's carefully re-examine the problem and the steps taken to solve it. Given the polynomial \( z^4 + z^2 + 1 |
| test:4516 | \frac{25 \sqrt{10}}{4} | NO_ANSWER | 1024 | True | 8\sqrt{10} | 815 | W->W | Let's carefully re-examine the steps and correct any potential errors. Given the parametric equations: \[ x(t |
| test:943 | 14 | 14 | 414 | False | 14 | 375 | C->C | The solution provided is correct. Let's verify the steps: 1. We know the arithmetic mean of \( A \), \( B \), |
| test:1324 | \frac{3}{16} | \frac{1}{2} | 678 | False | \frac{1}{2} | 613 | W->W | Let's carefully review the solution to identify any potential errors and ensure the logic is correct. The goa |
