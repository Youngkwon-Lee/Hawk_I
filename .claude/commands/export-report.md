# Export Report Command

분석 결과 내보내기 워크플로우

## PDF 리포트 생성 (Frontend)

```javascript
// frontend/src/lib/utils/exportPdf.ts 구현 필요
// jsPDF + html2canvas 사용

import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export async function exportToPdf(elementId: string, filename: string) {
  const element = document.getElementById(elementId);
  const canvas = await html2canvas(element);
  const pdf = new jsPDF();
  pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0);
  pdf.save(filename);
}
```

## CSV 메트릭 내보내기

```bash
cd backend

# 분석 결과를 CSV로 저장
python -c "
import json
import csv

# Load analysis result (from API or file)
result = {
    'video_id': 'abc123',
    'task_type': 'finger_tapping',
    'metrics': {
        'tapping_speed': 2.5,
        'amplitude_mean': 0.15,
        # ... more metrics
    },
    'updrs_score': 2
}

# Export to CSV
with open('analysis_result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    for key, val in result['metrics'].items():
        writer.writerow([key, val])
    writer.writerow(['updrs_score', result['updrs_score']])

print('Exported to analysis_result.csv')
"
```

## JSON 전체 데이터 내보내기

```bash
# API에서 결과 가져와서 저장
curl http://localhost:5000/api/analysis/result/{video_id} > result.json
```

## 구현 예정
- [ ] `/api/export/pdf/{video_id}` - 서버사이드 PDF 생성
- [ ] `/api/export/csv/{video_id}` - 메트릭 CSV
- [ ] `/api/export/fhir/{video_id}` - FHIR 표준 포맷
