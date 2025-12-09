"""
Analysis History Route
Provides API to list and filter past analysis results
"""

from flask import Blueprint, jsonify, request, current_app
import os
import json
from datetime import datetime
from pathlib import Path

bp = Blueprint('history', __name__, url_prefix='/api/history')


def parse_result_file(filepath: str) -> dict:
    """Parse a result JSON file and extract key information"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get file modification time as analysis date
        mtime = os.path.getmtime(filepath)
        analysis_date = datetime.fromtimestamp(mtime).isoformat()

        # Extract video_id from filename
        video_id = Path(filepath).stem.replace('_result', '')

        return {
            "video_id": video_id,
            "date": analysis_date,
            "task_type": data.get("video_type", "unknown"),
            "score": data.get("updrs_score", {}).get("total_score",
                     data.get("updrs_score", {}).get("score", None)),
            "severity": data.get("updrs_score", {}).get("severity", "Unknown"),
            "confidence": data.get("confidence", 0),
            "metrics": data.get("metrics", {}),
            "patient_id": data.get("patient_id", "anonymous"),
            "scoring_method": data.get("updrs_score", {}).get("method", "rule"),
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


@bp.route('/', methods=['GET'])
def get_history():
    """
    Get analysis history with optional filters

    Query params:
    - task_type: Filter by task type (finger_tapping, gait, etc.)
    - patient_id: Filter by patient ID
    - start_date: Filter by start date (ISO format)
    - end_date: Filter by end date (ISO format)
    - limit: Max number of results (default: 50)
    - offset: Pagination offset (default: 0)
    - sort: Sort order (date_desc, date_asc, score_desc, score_asc)
    """
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')

    # Get filter params
    task_type = request.args.get('task_type')
    patient_id = request.args.get('patient_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    sort = request.args.get('sort', 'date_desc')

    # Find all result files
    results = []
    result_pattern = Path(upload_folder).glob('*_result.json')

    for filepath in result_pattern:
        result = parse_result_file(str(filepath))
        if result:
            # Apply filters
            if task_type and result['task_type'] != task_type:
                continue
            if patient_id and result['patient_id'] != patient_id:
                continue
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    result_dt = datetime.fromisoformat(result['date'])
                    if result_dt < start_dt:
                        continue
                except:
                    pass
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    result_dt = datetime.fromisoformat(result['date'])
                    if result_dt > end_dt:
                        continue
                except:
                    pass

            results.append(result)

    # Sort results
    if sort == 'date_desc':
        results.sort(key=lambda x: x['date'], reverse=True)
    elif sort == 'date_asc':
        results.sort(key=lambda x: x['date'])
    elif sort == 'score_desc':
        results.sort(key=lambda x: x['score'] or 0, reverse=True)
    elif sort == 'score_asc':
        results.sort(key=lambda x: x['score'] or 0)

    # Get total before pagination
    total = len(results)

    # Apply pagination
    results = results[offset:offset + limit]

    return jsonify({
        "success": True,
        "data": {
            "items": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    })


@bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get aggregated statistics for history

    Query params:
    - task_type: Filter by task type
    - patient_id: Filter by patient ID
    """
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    task_type = request.args.get('task_type')
    patient_id = request.args.get('patient_id')

    # Find all result files
    results = []
    result_pattern = Path(upload_folder).glob('*_result.json')

    for filepath in result_pattern:
        result = parse_result_file(str(filepath))
        if result:
            if task_type and result['task_type'] != task_type:
                continue
            if patient_id and result['patient_id'] != patient_id:
                continue
            results.append(result)

    if not results:
        return jsonify({
            "success": True,
            "data": {
                "total_analyses": 0,
                "task_distribution": {},
                "score_distribution": {},
                "trend": []
            }
        })

    # Calculate statistics
    task_distribution = {}
    score_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for r in results:
        # Task distribution
        tt = r['task_type']
        task_distribution[tt] = task_distribution.get(tt, 0) + 1

        # Score distribution
        score = r['score']
        if score is not None:
            score_int = int(round(score))
            if score_int in score_distribution:
                score_distribution[score_int] += 1

    # Calculate trend (last 10 results sorted by date)
    results.sort(key=lambda x: x['date'])
    trend = [
        {
            "date": r['date'][:10],  # Just the date part
            "score": r['score'],
            "task_type": r['task_type']
        }
        for r in results[-20:]  # Last 20 for trend
        if r['score'] is not None
    ]

    # Average score
    scores = [r['score'] for r in results if r['score'] is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    return jsonify({
        "success": True,
        "data": {
            "total_analyses": len(results),
            "average_score": round(avg_score, 2) if avg_score else None,
            "task_distribution": task_distribution,
            "score_distribution": score_distribution,
            "trend": trend
        }
    })


@bp.route('/<video_id>', methods=['GET'])
def get_single_result(video_id: str):
    """Get a single analysis result by video ID"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    result_path = os.path.join(upload_folder, f"{video_id}_result.json")

    if not os.path.exists(result_path):
        return jsonify({
            "success": False,
            "error": "Analysis result not found"
        }), 404

    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return jsonify({
            "success": True,
            "data": data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route('/<video_id>', methods=['DELETE'])
def delete_result(video_id: str):
    """Delete an analysis result"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    result_path = os.path.join(upload_folder, f"{video_id}_result.json")

    if not os.path.exists(result_path):
        return jsonify({
            "success": False,
            "error": "Analysis result not found"
        }), 404

    try:
        os.remove(result_path)

        # Also try to remove related files
        related_patterns = [
            f"{video_id}.mp4",
            f"{video_id}_skeleton.mp4",
            f"{video_id}_heatmap.jpg",
            f"{video_id}_temporal.jpg",
        ]
        for pattern in related_patterns:
            path = os.path.join(upload_folder, pattern)
            if os.path.exists(path):
                os.remove(path)

        return jsonify({
            "success": True,
            "message": f"Deleted analysis {video_id}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
