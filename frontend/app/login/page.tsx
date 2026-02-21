'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { setUserTier, clearUserTier } from '@/lib/demoMode';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function LoginPage() {
  const router = useRouter();
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(true);

  // Check if already authenticated
  useEffect(() => {
    const stored = localStorage.getItem('windmar_api_key');
    if (stored) {
      fetch(`${API_BASE_URL}/api/demo/verify`, {
        method: 'POST',
        headers: { 'X-API-Key': stored },
      })
        .then(r => {
          if (r.ok) return r.json().then(data => { setUserTier(data.tier || 'demo'); router.replace('/'); });
          throw new Error();
        })
        .catch(() => {
          localStorage.removeItem('windmar_api_key');
          clearUserTier();
          setChecking(false);
        });
    } else {
      setChecking(false);
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!apiKey.trim()) return;

    setLoading(true);
    setError('');

    try {
      const resp = await fetch(`${API_BASE_URL}/api/demo/verify`, {
        method: 'POST',
        headers: { 'X-API-Key': apiKey.trim() },
      });
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      localStorage.setItem('windmar_api_key', apiKey.trim());
      setUserTier(data.tier || 'demo');
      router.replace('/');
    } catch {
      setError('Invalid licence key');
      setLoading(false);
    }
  };

  if (checking) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-slate-400 text-sm">Checking authentication...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <img src="/windmar-logo.png" alt="WindMar" className="w-16 h-16 rounded-2xl mb-4 inline-block" />
          <h1 className="text-xl font-bold text-white">WindMar</h1>
          <p className="text-sm text-slate-400 mt-1">Weather Routing &amp; Performance Analytics Demo</p>
        </div>

        {/* Login form */}
        <form onSubmit={handleSubmit} className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
          <label htmlFor="apiKey" className="block text-xs font-medium text-slate-400 mb-2">
            <svg className="inline w-3 h-3 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z" />
            </svg>
            Licence Key
          </label>
          <input
            id="apiKey"
            type="password"
            value={apiKey}
            onChange={(e) => { setApiKey(e.target.value); setError(''); }}
            placeholder="Enter licence key"
            autoFocus
            className="w-full px-3 py-2.5 bg-slate-900/50 border border-slate-600/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25"
          />

          {error && (
            <p className="mt-2 text-xs text-red-400">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading || !apiKey.trim()}
            className="w-full mt-4 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {loading ? 'Verifying...' : (
              <>
                Access Demo
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                </svg>
              </>
            )}
          </button>
        </form>

        <p className="text-center text-xs text-slate-500 mt-4">
          Ask for a key: <a href="mailto:contact@slmar.co" className="text-blue-400 hover:underline">contact@slmar.co</a>
        </p>
      </div>
    </div>
  );
}
