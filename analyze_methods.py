#!/usr/bin/env python3
"""Analyze method LOC ranges in build_graph.py"""
import re

with open('src/ingestion/build_graph.py', 'r') as f:
    lines = f.readlines()

# Find all method/function defs and their line numbers
defs = []
for i, line in enumerate(lines, 1):
    m = re.match(r'^(\s*)def (\w+)\(', line)
    if m:
        indent = len(m.group(1))
        name = m.group(2)
        defs.append((i, name, indent))

# Determine end of each method by finding the next def at same or lower indent,
# or end of class for class methods
for idx in range(len(defs)):
    start, name, indent = defs[idx]
    end = None
    for next_start, next_name, next_indent in defs[idx+1:]:
        if next_indent <= indent:
            end = next_start - 1
            break
    if end is None:
        end = len(lines)
    defs[idx] = (start, end, name, indent)

for start, end, name, indent in defs:
    loc = end - start + 1
    level = 'class' if indent > 0 else 'module'
    print(f'{name:45s} {start:5d}-{end:5d}  ({loc:4d} LOC)  [{level}]')
