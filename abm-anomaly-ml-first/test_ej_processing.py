from ml_analyzer import MLAnomalyAnalyzer
import os

# Read the actual EJ file
with open('/app/input/ABM250EJ_20250618_20250618.txt', 'r') as f:
    content = f.read()

analyzer = MLAnomalyAnalyzer()
sessions = analyzer.split_into_sessions(content, '/app/input/ABM250EJ_20250618_20250618.txt')

print(f'Total sessions found: {len(sessions)}')

# Look for the specific session at 21:07
target_session = None
for session in sessions:
    if session.start_time and session.start_time.hour == 21 and session.start_time.minute == 7:
        target_session = session
        break

if target_session:
    print(f'✓ Found target session at 21:07!')
    print(f'  Session ID: {target_session.session_id}')
    print(f'  Start time: {target_session.start_time}')
    print(f'  Contains PRIMARY CARD READER ACTIVATED: {"PRIMARY CARD READER ACTIVATED" in target_session.raw_text}')
    print(f'  First 200 chars: {target_session.raw_text[:200]}')
else:
    print('✗ Target session at 21:07 not found')
    
    # Show sessions around that time for debugging
    print('Sessions around 21:00-22:00:')
    for session in sessions:
        if session.start_time and 20 <= session.start_time.hour <= 22:
            print(f'  {session.start_time} - {session.session_id[:50]}...')
