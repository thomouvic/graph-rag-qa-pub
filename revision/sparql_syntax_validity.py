"""SPARQL syntax-validity rate per dataset/model.

For each row in the 6 SPARQL CoT CSVs (no-GW, paper-era), extract the SPARQL
query block from the model output (between SELECT and the matching closing
brace), and try to parse it with rdflib.plugins.sparql.parser.parseQuery.

Reports:
  - extraction rate: rows where we found *any* SPARQL-looking SELECT block
  - parse rate (of extracted): of those extracted, how many parse cleanly
  - parse rate (of all): rows whose extracted query parses (denominator = all rows)

The metric of interest for the rebuttal is parse-rate-of-all: it answers
'what fraction of model outputs produced syntactically valid SPARQL?'
"""
import csv
import os
import re
import sys

from rdflib.plugins.sparql import parser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_table5 import SPARQL_NO_GW

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_sparql(text):
    """Find first SELECT ... { ... } block. Return the full query string or None.

    Strategy: locate 'SELECT' (case-insensitive, whole word), then balance braces
    starting from the first '{' after it. If no balanced match found, return None.
    """
    if not text:
        return None
    m = re.search(r'\bSELECT\b', text, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    # Find first '{' after SELECT
    brace_start = text.find('{', start)
    if brace_start == -1:
        return None
    depth = 0
    end = None
    for i in range(brace_start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    query = text[start:end + 1]
    # Strip trailing junk after the closing brace would be re-added; we already
    # cut at the matching brace, so the query string is self-contained.
    return query.strip()


def normalize_for_parse(query):
    """rdflib's SPARQL parser is strict about IRIs/literals. The model output
    uses bare predicate names like 'name' or 'releasedBy' (no prefix, no <>).
    Wrap bare identifiers used as predicates in <> so the parser accepts them
    as IRIs. This measures *structural* syntax validity (well-formed triple
    patterns and braces), not predicate-vocabulary validity.
    """
    # Add a default prefix so bare names parse as PrefixedName under ':<name>'
    # is fragile; simpler: replace bare-word predicates in triple positions
    # with <name>. We do this conservatively with a regex over triple bodies.
    # Identify the WHERE block:
    m = re.search(r'\{(.*)\}', query, re.DOTALL)
    if not m:
        return query
    body = m.group(1)
    # In each triple, predicates are positions 2 of a 3-token triple.
    # We do a coarse pass: for any bare word (letters/underscore, no <> or "")
    # that doesn't start with ? or _: or "" or <, wrap it in <>.
    # We restrict to alphanumeric+underscore tokens that are not SPARQL keywords.
    KEYWORDS = {'a', 'FILTER', 'OPTIONAL', 'UNION', 'BIND', 'AS',
                'true', 'false', 'GRAPH', 'MINUS', 'SERVICE'}

    def wrap_bare_iri(match):
        tok = match.group(0)
        if tok in KEYWORDS:
            return tok
        return f'<{tok}>'

    # Match bare identifiers NOT starting with ? or " or < or _: or @
    # i.e. word characters that look like a bare predicate name
    body_fixed = re.sub(r'(?<![?<":\w@])[A-Za-z][A-Za-z0-9_]*(?!\s*[:>])',
                        wrap_bare_iri, body)
    # Re-assemble: replace inside the original query
    return query[:m.start() + 1] + body_fixed + query[m.end() - 1:]


def try_parse(query):
    """Return True if the query parses with rdflib's SPARQL parser, False else."""
    try:
        parser.parseQuery(query)
        return True
    except Exception:
        pass
    # Try after normalizing bare predicates
    try:
        parser.parseQuery(normalize_for_parse(query))
        return True
    except Exception:
        return False


def report(ds, model, csv_path, sample_failures=False):
    n_total = 0
    n_extracted = 0
    n_parsed = 0
    failed_examples = []
    with open(os.path.join(BASE, csv_path), 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            n_total += 1
            text = row.get('sparql_cot', '') or ''
            query = extract_sparql(text)
            if query is None:
                continue
            n_extracted += 1
            if try_parse(query):
                n_parsed += 1
            elif sample_failures and len(failed_examples) < 3:
                failed_examples.append(query[:300])
    return {
        'ds': ds, 'model': model,
        'n_total': n_total,
        'n_extracted': n_extracted,
        'n_parsed': n_parsed,
        'extract_rate': 100 * n_extracted / n_total if n_total else 0,
        'parse_of_extracted': 100 * n_parsed / n_extracted if n_extracted else 0,
        'parse_of_all': 100 * n_parsed / n_total if n_total else 0,
        'failed_examples': failed_examples,
    }


def main():
    print(f'{"Dataset":<18} {"Model":<5} {"N":>5} {"Extract%":>9} {"Parse/Extr%":>13} {"Parse/All%":>11}')
    print('-' * 70)
    results = []
    for (ds, model), path in sorted(SPARQL_NO_GW.items()):
        r = report(ds, model, path)
        results.append(r)
        print(f'{ds:<18} {model:<5} {r["n_total"]:>5} '
              f'{r["extract_rate"]:>8.1f}% {r["parse_of_extracted"]:>12.1f}% '
              f'{r["parse_of_all"]:>10.1f}%')
    print()
    # Per-model averages
    print('Per-model averages:')
    for model in ['8B', '70B']:
        rs = [r for r in results if r['model'] == model]
        avg_extract = sum(r['extract_rate'] for r in rs) / len(rs)
        avg_parse_all = sum(r['parse_of_all'] for r in rs) / len(rs)
        avg_parse_ext = sum(r['parse_of_extracted'] for r in rs) / len(rs)
        print(f'  {model}: extract={avg_extract:.1f}%  '
              f'parse-of-all={avg_parse_all:.1f}%  '
              f'parse-of-extracted={avg_parse_ext:.1f}%')


if __name__ == '__main__':
    main()
