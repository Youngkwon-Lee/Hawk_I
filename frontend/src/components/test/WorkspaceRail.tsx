import * as React from "react"
import { Check, Circle, FileCheck2, Leaf, UserRound } from "lucide-react"

interface WorkspaceSubject {
    id: string
    display_name: string
    sex?: string | null
    age?: number | null
    patient_id?: string | null
}

interface WorkspaceRailProps {
    selectedSubject?: WorkspaceSubject | null
    subjects?: WorkspaceSubject[]
    selectedSubjectId?: string
    onSelectSubject?: (subjectId: string) => void
}

export function WorkspaceRail({ selectedSubject, subjects = [], selectedSubjectId = "", onSelectSubject }: WorkspaceRailProps) {
    const [showSubjectPicker, setShowSubjectPicker] = React.useState(false)
    const patientName = selectedSubject?.display_name || "김하늘"
    const patientSexAge = selectedSubject
        ? `${selectedSubject.sex || ""}${selectedSubject.age ? ` · ${selectedSubject.age}세` : ""}`.trim()
        : "여 · 62세"
    const patientPid = selectedSubject?.patient_id || "00012345"

    return (
        <div className="flex h-full flex-col bg-card/55">
            <div className="border-b border-border px-5 py-5">
                <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-muted-foreground">현재 환자</p>
                    {subjects.length > 0 && onSelectSubject ? (
                        <button
                            type="button"
                            onClick={() => setShowSubjectPicker((open) => !open)}
                            className="rounded-md border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent"
                        >
                            변경
                        </button>
                    ) : null}
                </div>
                <div className="mt-4 flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full border border-primary/25 bg-primary/10 text-primary"><UserRound className="h-5 w-5" aria-hidden="true" /></div>
                    <div><p className="text-sm font-semibold">{patientName} <span className="text-xs font-normal text-muted-foreground">· 데모</span></p><p className="mt-1 text-xs text-muted-foreground">{patientSexAge}</p></div>
                </div>
                <dl className="mt-5 grid grid-cols-2 gap-x-3 gap-y-3 text-xs">
                    <div><dt className="text-muted-foreground">PID</dt><dd className="mt-1 font-medium text-foreground">{patientPid}</dd></div>
                    <div><dt className="text-muted-foreground">최근 검사</dt><dd className="mt-1 font-medium text-foreground">2025.05.12</dd></div>
                </dl>
                {showSubjectPicker && subjects.length > 0 && onSelectSubject ? (
                    <label className="mt-4 block space-y-1.5 text-xs text-muted-foreground">
                        환자 선택
                        <select
                            value={selectedSubjectId}
                            onChange={(event) => {
                                onSelectSubject(event.target.value)
                                setShowSubjectPicker(false)
                            }}
                            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
                        >
                            {subjects.map((subject) => <option key={subject.id} value={subject.id}>{subject.display_name}</option>)}
                        </select>
                    </label>
                ) : null}
            </div>

            <div className="flex-1 px-5 py-6">
                <p className="text-sm font-semibold text-foreground">진행 단계</p>
                <div className="relative mt-5 space-y-5 pl-1">
                    <div className="absolute left-[0.65rem] top-3 h-[8.5rem] w-px bg-border" aria-hidden="true" />
                    <ProgressItem icon={<Check className="h-3.5 w-3.5" />} title="검사 준비" detail="유형 선택 및 ParkiCheck 연결" complete />
                    <ProgressItem icon={<Circle className="h-3.5 w-3.5" />} title="데이터 업로드" detail="영상 파일 업로드" active />
                    <ProgressItem icon={<Circle className="h-3.5 w-3.5" />} title="분석 중" detail="AI 분석이 진행됩니다" />
                    <ProgressItem icon={<Circle className="h-3.5 w-3.5" />} title="결과 연결" detail="기록에 자동 저장됩니다" />
                </div>

                <div className="mt-9 rounded-xl border border-primary/15 bg-primary/[0.04] p-4">
                    <div className="flex items-start gap-2"><Leaf className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" /><p className="text-sm font-semibold text-primary">분석 결과는 기록에 자동 연결됩니다</p></div>
                    <p className="mt-3 text-xs leading-5 text-muted-foreground">결과는 환자 기록에 자동으로 연결되어 변화 추이를 한눈에 확인할 수 있습니다.</p>
                    <button className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline">기록 미리보기 <FileCheck2 className="h-3.5 w-3.5" aria-hidden="true" /></button>
                </div>
            </div>
        </div>
    )
}

function ProgressItem({ icon, title, detail, complete = false, active = false }: { icon: React.ReactNode; title: string; detail: string; complete?: boolean; active?: boolean }) {
    return (
        <div className="relative z-10 flex items-start gap-3">
            <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full border ${complete ? "border-emerald-400 bg-emerald-400 text-white" : active ? "border-primary bg-background text-primary" : "border-border bg-card text-muted-foreground"}`}>{icon}</span>
            <div className="min-w-0"><p className={`text-sm font-medium ${active ? "text-foreground" : complete ? "text-foreground" : "text-muted-foreground"}`}>{title}</p><p className="mt-1 text-xs leading-4 text-muted-foreground">{detail}</p></div>
        </div>
    )
}
