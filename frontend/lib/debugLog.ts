/**
 * Global debug log buffer for the in-app logging console.
 * Components call `debugLog(...)` and the DebugConsole component renders them.
 */

export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

export interface LogEntry {
  id: number;
  timestamp: Date;
  level: LogLevel;
  category: string;   // e.g. 'WAVE', 'ROUTE', 'API', 'WEATHER'
  message: string;
}

const MAX_ENTRIES = 200;
let _nextId = 1;
let _entries: LogEntry[] = [];
let _listeners: Array<() => void> = [];

export function debugLog(level: LogLevel, category: string, message: string) {
  const entry: LogEntry = {
    id: _nextId++,
    timestamp: new Date(),
    level,
    category: category.toUpperCase(),
    message,
  };
  _entries.push(entry);
  if (_entries.length > MAX_ENTRIES) {
    _entries = _entries.slice(-MAX_ENTRIES);
  }
  // Also mirror to browser console
  const prefix = `[${category.toUpperCase()}]`;
  if (level === 'error') console.error(prefix, message);
  else if (level === 'warn') console.warn(prefix, message);
  else console.log(prefix, message);

  // Notify subscribers
  for (const fn of _listeners) fn();
}

export function getLogEntries(): LogEntry[] {
  return _entries;
}

export function clearLog() {
  _entries = [];
  _nextId = 1;
  for (const fn of _listeners) fn();
}

export function subscribeLog(fn: () => void): () => void {
  _listeners.push(fn);
  return () => {
    _listeners = _listeners.filter(l => l !== fn);
  };
}
