/**
 * Analysis persistence using localStorage.
 * Provides CRUD operations for voyage analyses.
 */

import { Position, VoyageResponse } from './api';

export interface MonteCarloResult {
  n_simulations: number;
  eta: { p10: string; p50: string; p90: string };
  fuel_mt: { p10: number; p50: number; p90: number };
  total_time_hours: { p10: number; p50: number; p90: number };
  computation_time_ms: number;
}

export interface AnalysisEntry {
  id: string;
  routeName: string;
  waypoints: Position[];
  timestamp: string; // ISO — when analysis was run
  parameters: {
    calmSpeed: number;
    isLaden: boolean;
    useWeather: boolean;
    departureTime?: string;
  };
  result: VoyageResponse;
  monteCarlo?: MonteCarloResult;
}

const STORAGE_KEY = 'windmar_analyses';

function generateId(): string {
  return `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Get all analyses from localStorage, newest first.
 */
export function getAnalyses(): AnalysisEntry[] {
  if (typeof window === 'undefined') return [];

  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    const analyses: AnalysisEntry[] = JSON.parse(data);
    // Sort newest first
    return analyses.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  } catch (error) {
    console.error('Failed to load analyses:', error);
    return [];
  }
}

/**
 * Save a new analysis.
 */
export function saveAnalysis(
  routeName: string,
  waypoints: Position[],
  parameters: AnalysisEntry['parameters'],
  result: VoyageResponse,
): AnalysisEntry {
  const analyses = getAnalyses();

  const entry: AnalysisEntry = {
    id: generateId(),
    routeName,
    waypoints: [...waypoints],
    timestamp: new Date().toISOString(),
    parameters,
    result,
  };

  analyses.push(entry);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(analyses));

  return entry;
}

/**
 * Delete an analysis by ID.
 */
export function deleteAnalysis(id: string): boolean {
  const analyses = getAnalyses();
  const filtered = analyses.filter(a => a.id !== id);

  if (filtered.length === analyses.length) return false;

  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Update an analysis's Monte Carlo results.
 */
export function updateAnalysisMonteCarlo(id: string, mc: MonteCarloResult): AnalysisEntry | null {
  const analyses = getAnalyses();
  const index = analyses.findIndex(a => a.id === id);

  if (index === -1) return null;

  analyses[index] = {
    ...analyses[index],
    monteCarlo: mc,
  };

  localStorage.setItem(STORAGE_KEY, JSON.stringify(analyses));
  return analyses[index];
}
