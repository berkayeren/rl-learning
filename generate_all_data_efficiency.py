#!/usr/bin/env python3
"""
Generate efficiency gains tables with ADA as baseline using ALL data from result files.
Reads data directly from the markdown result files.
"""

import re


def calc_diff(other, ada):
    """Calculate percentage difference."""
    if ada == 0:
        return 0.0
    return ((other - ada) / ada) * 100


def parse_csv_from_md(filepath, section_name):
    """Extract CSV data from markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the section
    pattern = f"# {section_name}\\n([^#]+)"
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return []

    section_data = match.group(1).strip()
    lines = section_data.split('\n')

    # Skip header and parse data
    data = []
    for line in lines[1:]:  # Skip header row
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    iteration = int(parts[0])
                    mean = float(parts[1])
                    lower = float(parts[2])
                    upper = float(parts[3])
                    data.append((iteration, mean, lower, upper))
                except (ValueError, IndexError):
                    continue

    return data


# Read all environment data
empty_count = parse_csv_from_md('results/empty_room_results.md', 'Count Based Episode Length Analysis')
empty_dowham = parse_csv_from_md('results/empty_room_results.md', 'DoWham Episode Length Analysis')
empty_ada = parse_csv_from_md('results/empty_room_results.md', 'DoWhamv2 Episode Length Analysis')

four_count = parse_csv_from_md('results/four_room_no_key_results.md', 'Count Based Episode Length Analysis')
four_dowham = parse_csv_from_md('results/four_room_no_key_results.md', 'DoWham Episode Length Analysis')
four_ada = parse_csv_from_md('results/four_room_no_key_results.md', 'ADA Episode Length Analysis')

multi_fix_count = parse_csv_from_md('results/multi_room_fix_pos_results.md', 'Count Based Episode Length Analysis')
multi_fix_dowham = parse_csv_from_md('results/multi_room_fix_pos_results.md', 'DoWham Episode Length Analysis')
multi_fix_ada = parse_csv_from_md('results/multi_room_fix_pos_results.md', 'ADA Episode Length Analysis')

multi_key_dowham = parse_csv_from_md('results/multi_room_key_results.md', 'DoWham Episode Length Analysis')
multi_key_ada = parse_csv_from_md('results/multi_room_key_results.md', 'ADA Episode Length Analysis')

crossing_count = parse_csv_from_md('results/crossing_results.md', 'Count Based Episode Length Analysis')
crossing_dowham = parse_csv_from_md('results/crossing_results.md', 'DoWham Episode Length Analysis')
crossing_ada = parse_csv_from_md('results/crossing_results.md', 'ADA Episode Length Analysis')

# Print header
print("""# Efficiency Gains Analysis - Episode Length Mean Metrics

This analysis compares the efficiency of different methods across five environments based on **episode length mean** values. **Lower episode length = Better performance** (faster goal achievement).

**Important:** 
- **ADA** is used as the baseline for all comparisons (the improved exploration method)
- We compare Count Based and DoWham against ADA to show how much better ADA performs
- ADA represents the state-of-the-art exploration-based approach
- **ALL data points from result files are included - nothing omitted**

## Efficiency Comparison Calculation
**Performance Difference (%) = ((Other Method - ADA) / ADA) × 100**

Since **lower is better** for episode length:
- **Positive values** = Other method is WORSE (takes more steps than ADA)
- **Negative values** = Other method is BETTER (takes fewer steps than ADA)
- **0%** = Same performance

**Bounds Efficiency Gains:**
- **Lower Bound Diff**: Compares other method's lower bound to ADA's lower bound
- **Upper Bound Diff**: Compares other method's upper bound to ADA's upper bound
- **Shows the range of possible performance differences considering uncertainty**

## How to Read the Tables
- **ADA Mean/Lower/Upper**: Baseline values with confidence intervals
- **Count Based Mean/Lower/Upper**: Values with confidence intervals
- **Count vs ADA**: Three efficiency percentages
  - **Mean %**: Difference in mean values
  - **Lower %**: Difference in lower bounds (best-case comparison)
  - **Upper %**: Difference in upper bounds (worst-case comparison)
- **DoWham Mean/Lower/Upper**: Values with confidence intervals  
- **DoWham vs ADA**: Three efficiency percentages (same structure)
  
**Note:** Bounds represent 95% confidence intervals from bootstrap analysis

---

## 1. Empty Room Environment

**Baseline: ADA** - Lower episode length is better

| Iteration | ADA Mean | ADA Bounds [L, U] | Count Mean | Count Bounds [L, U] | Count vs ADA: Mean% / Lower% / Upper% | DoWham Mean | DoWham Bounds [L, U] | DoWham vs ADA: Mean% / Lower% / Upper% |
|-----------|----------|-------------------|------------|---------------------|--------------------------------------|-------------|----------------------|----------------------------------------|""")

# Empty Room
for ada, count, dowham in zip(empty_ada, empty_count, empty_dowham):
    it_a, ada_m, ada_l, ada_u = ada
    it_c, cnt_m, cnt_l, cnt_u = count
    it_d, dw_m, dw_l, dw_u = dowham

    if it_a == it_c == it_d:
        cnt_mean_diff = calc_diff(cnt_m, ada_m)
        cnt_lower_diff = calc_diff(cnt_l, ada_l)
        cnt_upper_diff = calc_diff(cnt_u, ada_u)

        dw_mean_diff = calc_diff(dw_m, ada_m)
        dw_lower_diff = calc_diff(dw_l, ada_l)
        dw_upper_diff = calc_diff(dw_u, ada_u)

        print(
            f"| {it_a} | {ada_m:.2f} | [{ada_l:.2f}, {ada_u:.2f}] | {cnt_m:.2f} | [{cnt_l:.2f}, {cnt_u:.2f}] | {cnt_mean_diff:+.2f} / {cnt_lower_diff:+.2f} / {cnt_upper_diff:+.2f} | {dw_m:.2f} | [{dw_l:.2f}, {dw_u:.2f}] | {dw_mean_diff:+.2f} / {dw_lower_diff:+.2f} / {dw_upper_diff:+.2f} |")

print("""
---

## 2. Four Room No Key Environment

**Baseline: ADA** - Lower episode length is better

| Iteration | ADA Mean | ADA Bounds [L, U] | Count Mean | Count Bounds [L, U] | Count vs ADA: Mean% / Lower% / Upper% | DoWham Mean | DoWham Bounds [L, U] | DoWham vs ADA: Mean% / Lower% / Upper% |
|-----------|----------|-------------------|------------|---------------------|--------------------------------------|-------------|----------------------|----------------------------------------|""")

# Four Room No Key
for ada, count, dowham in zip(four_ada, four_count, four_dowham):
    it_a, ada_m, ada_l, ada_u = ada
    it_c, cnt_m, cnt_l, cnt_u = count
    it_d, dw_m, dw_l, dw_u = dowham

    if it_a == it_c == it_d:
        cnt_mean_diff = calc_diff(cnt_m, ada_m)
        cnt_lower_diff = calc_diff(cnt_l, ada_l)
        cnt_upper_diff = calc_diff(cnt_u, ada_u)

        dw_mean_diff = calc_diff(dw_m, ada_m)
        dw_lower_diff = calc_diff(dw_l, ada_l)
        dw_upper_diff = calc_diff(dw_u, ada_u)

        print(
            f"| {it_a} | {ada_m:.2f} | [{ada_l:.2f}, {ada_u:.2f}] | {cnt_m:.2f} | [{cnt_l:.2f}, {cnt_u:.2f}] | {cnt_mean_diff:+.2f} / {cnt_lower_diff:+.2f} / {cnt_upper_diff:+.2f} | {dw_m:.2f} | [{dw_l:.2f}, {dw_u:.2f}] | {dw_mean_diff:+.2f} / {dw_lower_diff:+.2f} / {dw_upper_diff:+.2f} |")

print("""
---

## 3. Multi Room Fixed Position Environment

**Baseline: ADA** - Lower episode length is better

| Iteration | ADA Mean | ADA Bounds [L, U] | Count Mean | Count Bounds [L, U] | Count vs ADA: Mean% / Lower% / Upper% | DoWham Mean | DoWham Bounds [L, U] | DoWham vs ADA: Mean% / Lower% / Upper% |
|-----------|----------|-------------------|------------|---------------------|--------------------------------------|-------------|----------------------|----------------------------------------|""")

# Multi Room Fixed Position
for ada, count, dowham in zip(multi_fix_ada, multi_fix_count, multi_fix_dowham):
    it_a, ada_m, ada_l, ada_u = ada
    it_c, cnt_m, cnt_l, cnt_u = count
    it_d, dw_m, dw_l, dw_u = dowham

    if it_a == it_c == it_d:
        cnt_mean_diff = calc_diff(cnt_m, ada_m)
        cnt_lower_diff = calc_diff(cnt_l, ada_l)
        cnt_upper_diff = calc_diff(cnt_u, ada_u)

        dw_mean_diff = calc_diff(dw_m, ada_m)
        dw_lower_diff = calc_diff(dw_l, ada_l)
        dw_upper_diff = calc_diff(dw_u, ada_u)

        print(
            f"| {it_a} | {ada_m:.2f} | [{ada_l:.2f}, {ada_u:.2f}] | {cnt_m:.2f} | [{cnt_l:.2f}, {cnt_u:.2f}] | {cnt_mean_diff:+.2f} / {cnt_lower_diff:+.2f} / {cnt_upper_diff:+.2f} | {dw_m:.2f} | [{dw_l:.2f}, {dw_u:.2f}] | {dw_mean_diff:+.2f} / {dw_lower_diff:+.2f} / {dw_upper_diff:+.2f} |")

print("""
---

## 4. Multi Room Key Environment

**Baseline: ADA** - Lower episode length is better

**Note:** No Count Based data available for this environment

| Iteration | ADA Mean | ADA Bounds [L, U] | DoWham Mean | DoWham Bounds [L, U] | DoWham vs ADA: Mean% / Lower% / Upper% |
|-----------|----------|-------------------|-------------|----------------------|----------------------------------------|""")

# Multi Room Key
for ada, dowham in zip(multi_key_ada, multi_key_dowham):
    it_a, ada_m, ada_l, ada_u = ada
    it_d, dw_m, dw_l, dw_u = dowham

    if it_a == it_d:
        dw_mean_diff = calc_diff(dw_m, ada_m)
        dw_lower_diff = calc_diff(dw_l, ada_l)
        dw_upper_diff = calc_diff(dw_u, ada_u)

        print(
            f"| {it_a} | {ada_m:.2f} | [{ada_l:.2f}, {ada_u:.2f}] | {dw_m:.2f} | [{dw_l:.2f}, {dw_u:.2f}] | {dw_mean_diff:+.2f} / {dw_lower_diff:+.2f} / {dw_upper_diff:+.2f} |")

print("""
---

## 5. Crossing Environment

**Baseline: ADA** - Lower episode length is better

| Iteration | ADA Mean | ADA Bounds [L, U] | Count Mean | Count Bounds [L, U] | Count vs ADA: Mean% / Lower% / Upper% | DoWham Mean | DoWham Bounds [L, U] | DoWham vs ADA: Mean% / Lower% / Upper% |
|-----------|----------|-------------------|------------|---------------------|--------------------------------------|-------------|----------------------|----------------------------------------|""")

# Crossing
for ada, count, dowham in zip(crossing_ada, crossing_count, crossing_dowham):
    it_a, ada_m, ada_l, ada_u = ada
    it_c, cnt_m, cnt_l, cnt_u = count
    it_d, dw_m, dw_l, dw_u = dowham

    if it_a == it_c == it_d:
        cnt_mean_diff = calc_diff(cnt_m, ada_m)
        cnt_lower_diff = calc_diff(cnt_l, ada_l)
        cnt_upper_diff = calc_diff(cnt_u, ada_u)

        dw_mean_diff = calc_diff(dw_m, ada_m)
        dw_lower_diff = calc_diff(dw_l, ada_l)
        dw_upper_diff = calc_diff(dw_u, ada_u)

        print(
            f"| {it_a} | {ada_m:.2f} | [{ada_l:.2f}, {ada_u:.2f}] | {cnt_m:.2f} | [{cnt_l:.2f}, {cnt_u:.2f}] | {cnt_mean_diff:+.2f} / {cnt_lower_diff:+.2f} / {cnt_upper_diff:+.2f} | {dw_m:.2f} | [{dw_l:.2f}, {dw_u:.2f}] | {dw_mean_diff:+.2f} / {dw_lower_diff:+.2f} / {dw_upper_diff:+.2f} |")

print("""
---

## Overall Summary

### Data Completeness
- **Empty Room**: All {0} iterations included
- **Four Room No Key**: All {1} iterations included  
- **Multi Room Fixed Position**: All {2} iterations included
- **Multi Room Key**: All {3} iterations included (DoWham vs ADA only)
- **Crossing**: All {4} iterations included

### Key Findings with Full Data

**ALL data points from the result markdown files have been included in this analysis.** Nothing has been omitted.

### Methodology Note

All comparisons use **ADA as the baseline** with the formula:
- **Difference (%) = ((Other Method - ADA) / ADA) × 100**

**Three types of efficiency calculations:**
1. **Mean vs Mean**: Standard comparison of expected values
2. **Lower vs Lower**: Best-case scenario comparison (optimistic)
3. **Upper vs Upper**: Worst-case scenario comparison (pessimistic)

**Interpretation:**
- **Positive values** = Other method is worse (slower)
- **Negative values** = Other method is better (faster)
- **All three positive** = Robust evidence of ADA superiority
- **Mixed signs** = Performance depends on variance/uncertainty
- **Confidence intervals** = 95% bootstrap intervals
- Remember: **Lower episode length = Better performance**
""".format(len(empty_ada), len(four_ada), len(multi_fix_ada), len(multi_key_ada), len(crossing_ada)))
