'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { debugLog, getLogEntries, clearLog, subscribeLog, LogEntry, LogLevel } from '@/lib/debugLog';

const LEVEL_COLORS: Record<LogLevel, string> = {
  info: 'text-sky-400',
  warn: 'text-amber-400',
  error: 'text-red-400',
  debug: 'text-gray-400',
};

const LEVEL_BG: Record<LogLevel, string> = {
  info: 'bg-sky-500/10',
  warn: 'bg-amber-500/10',
  error: 'bg-red-500/10',
  debug: 'bg-transparent',
};

function formatTime(d: Date): string {
  return d.toLocaleTimeString('en-GB', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

export default function DebugConsole() {
  const [isOpen, setIsOpen] = useState(false);
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<string>('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  const refresh = useCallback(() => {
    setEntries([...getLogEntries()]);
  }, []);

  useEffect(() => {
    refresh();
    return subscribeLog(refresh);
  }, [refresh]);

  // Subscribe to backend SSE log stream
  useEffect(() => {
    let es: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003';
      es = new EventSource(`${apiBase}/api/logs/stream`);
      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          const level = (['error', 'warn', 'info', 'debug'].includes(data.level) ? data.level : 'info') as LogLevel;
          debugLog(level, 'SERVER', data.msg || '');
        } catch { /* ignore parse errors */ }
      };
      es.onerror = () => {
        es?.close();
        retryTimer = setTimeout(connect, 5000);
      };
    };
    connect();

    return () => {
      es?.close();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40;
  };

  const filtered = filter
    ? entries.filter(e =>
        e.category.includes(filter.toUpperCase()) ||
        e.message.toLowerCase().includes(filter.toLowerCase())
      )
    : entries;

  const errorCount = entries.filter(e => e.level === 'error').length;
  const warnCount = entries.filter(e => e.level === 'warn').length;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[9999] pointer-events-none">
      {/* Toggle button */}
      <div className="pointer-events-auto flex justify-end px-2 pb-0">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="px-3 py-1 text-xs font-mono rounded-t-md bg-gray-900/95 border border-b-0 border-gray-600/50 text-gray-300 hover:text-white flex items-center gap-2"
        >
          <span>Console</span>
          {errorCount > 0 && <span className="text-red-400 font-bold">{errorCount}E</span>}
          {warnCount > 0 && <span className="text-amber-400">{warnCount}W</span>}
          <span className="text-gray-500">{entries.length}</span>
        </button>
      </div>

      {/* Console panel */}
      {isOpen && (
        <div className="pointer-events-auto bg-gray-900/95 backdrop-blur-sm border-t border-gray-600/50 max-h-[300px] flex flex-col">
          {/* Toolbar */}
          <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-700/50 text-xs">
            <input
              type="text"
              placeholder="Filter (category or text)..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="bg-gray-800 text-gray-200 px-2 py-0.5 rounded border border-gray-600/40 w-48 font-mono text-[11px]"
            />
            <button
              onClick={() => { clearLog(); refresh(); }}
              className="px-2 py-0.5 text-gray-400 hover:text-white border border-gray-600/40 rounded"
            >
              Clear
            </button>
            <div className="flex-1" />
            <span className="text-gray-500 font-mono">{filtered.length} entries</span>
            <button
              onClick={() => setIsOpen(false)}
              className="px-1 text-gray-400 hover:text-white text-sm"
            >
              x
            </button>
          </div>

          {/* Log entries */}
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="flex-1 overflow-y-auto overflow-x-hidden font-mono text-[11px] leading-[18px] px-1"
          >
            {filtered.length === 0 ? (
              <div className="text-gray-500 p-4 text-center">No log entries</div>
            ) : (
              filtered.map(entry => (
                <div key={entry.id} className={`flex gap-2 px-2 py-[1px] ${LEVEL_BG[entry.level]} hover:bg-white/5`}>
                  <span className="text-gray-500 shrink-0 w-[85px]">{formatTime(entry.timestamp)}</span>
                  <span className={`shrink-0 w-[50px] ${LEVEL_COLORS[entry.level]}`}>
                    {entry.level.toUpperCase().padEnd(5)}
                  </span>
                  <span className="text-cyan-400 shrink-0 w-[70px] truncate">{entry.category}</span>
                  <span className="text-gray-200 break-all">{entry.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
