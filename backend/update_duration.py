with open('agents/vision_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('max_duration=5.0', 'max_duration=7.0')
content = content.replace('Analyze only first 5 seconds', 'Analyze only first 7 seconds')

with open('agents/vision_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! max_duration changed from 5.0 to 7.0 seconds")
