import json
from pathlib import Path

tl = json.loads(Path('output/compiled_timeline.json').read_text())
print('Event count:', len(tl))

event_types = {}
for e in tl:
    t = e.get('event_type', '?')
    event_types[t] = event_types.get(t, 0) + 1
print('Event type counts:', event_types)

seen = set()
for e in tl:
    t = e.get('event_type', '?')
    if t not in seen:
        seen.add(t)
        s = e.get('start_time', '?')
        en = e.get('end_time', '?')
        ez = e.get('easing', '?')
        print('FIRST OF TYPE=' + str(t) + '  start=' + str(s) + '  end=' + str(en) + '  easing=' + str(ez))

print('Last event start=' + str(tl[-1].get('start_time')) + ' end=' + str(tl[-1].get('end_time')) + ' type=' + str(tl[-1].get('event_type')))
print('Total duration (last end_time):', max(e.get('end_time', 0) for e in tl))

# Show all unique phase markers
print('\n--- All phase events (non-intro/non-drift) ---')
for e in tl:
    t = e.get('event_type', '?')
    if t not in ('intro_reveal', 'drift', 'logo_morph', 'logo_hold', 'logo_dissolve'):
        s = e.get('start_time', '?')
        en = e.get('end_time', '?')
        print('  type=' + str(t) + ' start=' + str(s) + ' end=' + str(en))

# Logo-related events
print('\n--- Logo events ---')
for e in tl:
    t = e.get('event_type', '?')
    if 'logo' in t.lower():
        s = e.get('start_time', '?')
        en = e.get('end_time', '?')
        logo = e.get('logo_index', e.get('metadata', {}).get('logo_index', '?'))
        print('  type=' + str(t) + ' logo=' + str(logo) + ' start=' + str(s) + ' end=' + str(en) + ' easing=' + str(e.get('easing','?')))
