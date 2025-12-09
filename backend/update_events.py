with open('routes/analyze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update import to include detect_events
old_import = 'from services.visualization_data_generator import generate_visualization_data'
new_import = 'from services.visualization_data_generator import generate_visualization_data, detect_events'

content = content.replace(old_import, new_import)

# 2. Add event detection after visualization_data generation
old_viz = '''        visualization_data = generate_visualization_data(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps
        )

        response = {'''

new_viz = '''        visualization_data = generate_visualization_data(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps
        )

        # Detect clinically relevant events
        event_task_type = "finger_tapping" if "finger" in ctx.task_type.lower() or "tapping" in ctx.task_type.lower() else "gait"
        detected_events = detect_events(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps,
            task_type=event_task_type
        )

        response = {'''

content = content.replace(old_viz, new_viz)

# 3. Replace empty events with detected_events
old_events = '"events": [], # Event detector not in agents yet'
new_events = '"events": detected_events,'

content = content.replace(old_events, new_events)

with open('routes/analyze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! analyze.py updated with event detection")
