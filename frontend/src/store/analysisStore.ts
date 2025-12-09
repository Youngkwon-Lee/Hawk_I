/**
 * Zustand Store for Analysis State Management
 * Replaces sessionStorage-based state passing between pages
 */
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { AnalysisResult } from '@/lib/services/api'

interface AnalysisState {
  // Current analysis result
  result: AnalysisResult | null

  // Analysis status
  isAnalyzing: boolean
  videoId: string | null
  progress: number

  // Error state
  error: string | null

  // Actions
  setResult: (result: AnalysisResult | null) => void
  setAnalyzing: (isAnalyzing: boolean, videoId?: string) => void
  setProgress: (progress: number) => void
  setError: (error: string | null) => void
  clearResult: () => void
  reset: () => void
}

const initialState = {
  result: null,
  isAnalyzing: false,
  videoId: null,
  progress: 0,
  error: null,
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set) => ({
      ...initialState,

      setResult: (result) => set({
        result,
        isAnalyzing: false,
        progress: 100,
        error: null
      }),

      setAnalyzing: (isAnalyzing, videoId) => set({
        isAnalyzing,
        videoId: videoId || null,
        progress: isAnalyzing ? 0 : 100,
        error: null
      }),

      setProgress: (progress) => set({ progress }),

      setError: (error) => set({
        error,
        isAnalyzing: false
      }),

      clearResult: () => set({
        result: null,
        videoId: null,
        progress: 0
      }),

      reset: () => set(initialState),
    }),
    {
      name: 'hawkeye-analysis-storage',
      storage: createJSONStorage(() => sessionStorage),
      // Only persist result and videoId, not transient states
      partialize: (state) => ({
        result: state.result,
        videoId: state.videoId
      }),
    }
  )
)

// Selector hooks for specific state slices
export const useAnalysisResult = () => useAnalysisStore((state) => state.result)
export const useIsAnalyzing = () => useAnalysisStore((state) => state.isAnalyzing)
export const useAnalysisProgress = () => useAnalysisStore((state) => state.progress)
export const useAnalysisError = () => useAnalysisStore((state) => state.error)
