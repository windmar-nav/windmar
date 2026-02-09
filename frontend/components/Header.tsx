'use client';

import { useState, useRef, useEffect } from 'react';
import { Ship, Gauge, Shield } from 'lucide-react';
import Link from 'next/link';
import VoyageDropdown from '@/components/VoyageDropdown';
import RegulationsDropdown from '@/components/RegulationsDropdown';

type DropdownId = 'voyage' | 'regulations' | null;

export default function Header() {
  const [openDropdown, setOpenDropdown] = useState<DropdownId>(null);
  const headerRef = useRef<HTMLElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (headerRef.current && !headerRef.current.contains(e.target as Node)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const toggle = (id: DropdownId) => {
    setOpenDropdown((prev) => (prev === id ? null : id));
  };

  return (
    <header ref={headerRef} className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/10">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Logo + subtitle */}
          <Link href="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <Ship className="w-8 h-8 text-primary-400 group-hover:text-primary-300 transition-colors" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-ocean-400 rounded-full animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl font-bold maritime-gradient-text">
                WINDMAR
              </h1>
              <p className="text-xs text-gray-400">Marine Route Analysis</p>
            </div>
          </Link>

          {/* Icon buttons */}
          <div className="flex items-center space-x-1">
            {/* Vessel — navigates to /vessel */}
            <Link
              href="/vessel"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
              title="Vessel"
            >
              <Ship className="w-5 h-5" />
              <span className="text-sm font-medium hidden sm:inline">Vessel</span>
            </Link>

            {/* Voyage — dropdown */}
            <div className="relative">
              <button
                onClick={() => toggle('voyage')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  openDropdown === 'voyage'
                    ? 'text-primary-400 bg-primary-500/10'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
                title="Voyage Parameters"
              >
                <Gauge className="w-5 h-5" />
                <span className="text-sm font-medium hidden sm:inline">Voyage</span>
              </button>
              {openDropdown === 'voyage' && <VoyageDropdown />}
            </div>

            {/* Regulations — dropdown */}
            <div className="relative">
              <button
                onClick={() => toggle('regulations')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  openDropdown === 'regulations'
                    ? 'text-primary-400 bg-primary-500/10'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
                title="Regulations"
              >
                <Shield className="w-5 h-5" />
                <span className="text-sm font-medium hidden sm:inline">Regulations</span>
              </button>
              {openDropdown === 'regulations' && <RegulationsDropdown />}
            </div>

            {/* Separator */}
            <div className="w-px h-6 bg-white/10 mx-2" />

            {/* Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm text-gray-300">Online</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
