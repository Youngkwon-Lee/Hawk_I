import os

# Update all chart components to conditionally show mock data label

components = [
    'src/components/dashboard/JointAngleChart.tsx',
    'src/components/dashboard/SymmetryChart.tsx',
    'src/components/dashboard/GaitCycleChart.tsx',
    'src/components/dashboard/SpeedProfileChart.tsx'
]

for comp in components:
    with open(comp, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add isUsingMockData variable
    if 'const chartData = data ||' in content:
        old_line = 'const chartData = data ||'
        # Find the full line
        for line in content.split('\n'):
            if old_line in line:
                # Extract the mock data variable name
                parts = line.split('||')
                if len(parts) == 2:
                    mock_var = parts[1].strip()
                    new_lines = f'''const isUsingMockData = !data || data.length === 0
    const chartData = data && data.length > 0 ? data :'''
                    content = content.replace(old_line, new_lines)
                    break

    # Update the label to be conditional
    old_label = '<p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>'
    new_label = '{isUsingMockData && <p className="text-xs text-muted-foreground text-center mt-2">(임시 데이터)</p>}'
    content = content.replace(old_label, new_label)

    with open(comp, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Updated {comp}")

print("\nDone!")
