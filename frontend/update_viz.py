import re

with open('src/app/result/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Add imports for new visualization components
old_imports = 'import { ReasoningLogViewer } from "@/components/dashboard/ReasoningLogViewer"'

new_imports = '''import { ReasoningLogViewer } from "@/components/dashboard/ReasoningLogViewer"
import { JointAngleChart } from "@/components/dashboard/JointAngleChart"
import { SymmetryChart } from "@/components/dashboard/SymmetryChart"
import { GaitCycleChart } from "@/components/dashboard/GaitCycleChart"
import { SpeedProfileChart } from "@/components/dashboard/SpeedProfileChart"'''

content = content.replace(old_imports, new_imports)

# Find the visualizations tab and add new components after the existing cards
# We'll insert the new charts after the existing grid

old_closing = '''                        </div>
                    </div>
                )}

                {activeTab === "timeline"'''

new_closing = '''                        </div>

                        {/* Row 2: Chart-based visualizations */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <JointAngleChart />
                            <SymmetryChart />
                        </div>

                        {/* Row 3: Gait cycle and speed profile */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <GaitCycleChart />
                            <SpeedProfileChart />
                        </div>
                    </div>
                )}

                {activeTab === "timeline"'''

content = content.replace(old_closing, new_closing)

with open('src/app/result/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
