"""Deterministic formatting of genuine observations."""
import re
from datetime import datetime,timedelta

def observation_text(o):
    return 'DATE: ' + o['date'] + '; ' + o['speaker'] + ' said: ' + o['body']

def temporal_notes(o):
    try:
        date = datetime.strptime(o['date'].split(' on ')[-1], '%d %B, %Y')
    except ValueError:
        return ''
    text = o['body'].lower()
    notes = []
    for (term, days) in (('yesterday', 1), ('the day before yesterday', 2), ('tomorrow', -1)):
        if re.search('\\b' + term + '\\b', text) and (not (term == 'yesterday' and 'the day before yesterday' in text)):
            notes.append(term + ' = ' + (date - timedelta(days=days)).strftime('%-d %B %Y'))
    if 'last year' in text:
        notes.append('last year = ' + str(date.year - 1))
    return '; '.join(notes)
