import re

with open('routes/analyze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import
old_import = 'from services.progress_tracker import init_analysis, update_step, complete_analysis, fail_analysis'
new_import = '''from services.progress_tracker import init_analysis, update_step, complete_analysis, fail_analysis
from services.visualization_data_generator import generate_visualization_data'''

content = content.replace(old_import, new_import)

# 2. Add visualization_data generation before response
old_response_start = '''        response = {
            "success": True,'''

new_response_start = '''        # Generate visualization data for charts
        fps = ctx.vision_meta.get("fps", 30.0)
        gait_analysis = ctx.vision_meta.get("gait_cycle_analysis")  # If available
        visualization_data = generate_visualization_data(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps
        )

        response = {
            "success": True,'''

content = content.replace(old_response_start, new_response_start)

# 3. Add visualization_data to response
old_viz_urls = '''            "visualization_urls": {
                "heatmap": f"/files/{os.path.basename(heatmap_path)}" if heatmap_path else None,
                "temporal_map": f"/files/{os.path.basename(temporal_path)}" if temporal_path else None,
                "attention_map": None
            }
        }'''

new_viz_urls = '''            "visualization_urls": {
                "heatmap": f"/files/{os.path.basename(heatmap_path)}" if heatmap_path else None,
                "temporal_map": f"/files/{os.path.basename(temporal_path)}" if temporal_path else None,
                "attention_map": None
            },
            "visualization_data": visualization_data
        }'''

content = content.replace(old_viz_urls, new_viz_urls)

with open('routes/analyze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
