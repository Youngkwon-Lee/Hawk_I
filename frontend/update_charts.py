import re

with open('src/app/result/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the visualization section and update component calls to pass data
old_charts = '''                        {/* Row 2: Chart-based visualizations */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <JointAngleChart />
                            <SymmetryChart />
                        </div>

                        {/* Row 3: Gait cycle and speed profile */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <GaitCycleChart />
                            <SpeedProfileChart />
                        </div>'''

new_charts = '''                        {/* Row 2: Chart-based visualizations */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <JointAngleChart data={analysisResult.visualization_data?.joint_angles} />
                            <SymmetryChart data={analysisResult.visualization_data?.symmetry} />
                        </div>

                        {/* Row 3: Gait cycle and speed profile */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <GaitCycleChart data={analysisResult.visualization_data?.gait_cycles} />
                            <SpeedProfileChart data={analysisResult.visualization_data?.speed_profile} />
                        </div>'''

content = content.replace(old_charts, new_charts)

with open('src/app/result/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
