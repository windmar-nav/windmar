'use client';

import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { DEMO_MODE } from '@/lib/demoMode';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface AuthGateProps {
  children: React.ReactNode;
}

export function AuthGate({ children }: AuthGateProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [authorized, setAuthorized] = useState(!DEMO_MODE);
  const [checking, setChecking] = useState(DEMO_MODE);

  useEffect(() => {
    if (!DEMO_MODE) return;
    if (pathname === '/login') {
      setChecking(false);
      setAuthorized(true); // Let login page render
      return;
    }

    const stored = localStorage.getItem('windmar_api_key');
    if (!stored) {
      router.replace('/login');
      return;
    }

    fetch(`${API_BASE_URL}/api/demo/verify`, {
      method: 'POST',
      headers: { 'X-API-Key': stored },
    })
      .then(r => {
        if (r.ok) {
          setAuthorized(true);
          setChecking(false);
        } else {
          localStorage.removeItem('windmar_api_key');
          router.replace('/login');
        }
      })
      .catch(() => {
        // If backend unreachable, allow access (don't lock out)
        setAuthorized(true);
        setChecking(false);
      });
  }, [pathname, router]);

  if (checking) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-slate-400 text-sm">Loading...</div>
      </div>
    );
  }

  if (!authorized) return null;

  return <>{children}</>;
}
