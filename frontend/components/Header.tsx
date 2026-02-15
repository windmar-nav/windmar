'use client';

import { useState, useRef, useEffect } from 'react';
import { Ship, Gauge, Shield, Map, LogOut, Cloud, BarChart3, ScrollText } from 'lucide-react';
import Link from 'next/link';
import VoyageDropdown from '@/components/VoyageDropdown';
import RegulationsDropdown from '@/components/RegulationsDropdown';
import { useVoyage } from '@/components/VoyageContext';

type DropdownId = 'voyage' | 'regulations' | null;

interface HeaderProps {
  onFitRoute?: () => void;
}

export default function Header({ onFitRoute }: HeaderProps) {
  const { viewMode, setViewMode } = useVoyage();
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

  const closeDropdown = () => setOpenDropdown(null);

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
            {/* Chart — navigates to / */}
            <Link
              href="/"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
              title="Chart"
            >
              <Map className="w-5 h-5" />
              <span className="text-sm font-medium hidden sm:inline">Chart</span>
            </Link>

            {/* Mode toggle: Weather / Analysis */}
            <div className="flex items-center bg-white/5 rounded-lg p-0.5">
              <button
                onClick={() => setViewMode('weather')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'weather'
                    ? 'bg-primary-500/20 text-primary-400'
                    : 'text-gray-400 hover:text-white'
                }`}
                title="Weather Viewer"
              >
                <Cloud className="w-4 h-4" />
                <span className="hidden sm:inline">Weather</span>
              </button>
              <button
                onClick={() => setViewMode('analysis')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'analysis'
                    ? 'bg-primary-500/20 text-primary-400'
                    : 'text-gray-400 hover:text-white'
                }`}
                title="Route Analysis"
              >
                <BarChart3 className="w-4 h-4" />
                <span className="hidden sm:inline">Analysis</span>
              </button>
            </div>

            {/* Vessel — navigates to /vessel */}
            <Link
              href="/vessel"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
              title="Vessel"
            >
              <Ship className="w-5 h-5" />
              <span className="text-sm font-medium hidden sm:inline">Vessel</span>
            </Link>

            {/* Engine Log — navigates to /engine-log */}
            <Link
              href="/engine-log"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
              title="Engine Log"
            >
              <ScrollText className="w-5 h-5" />
              <span className="text-sm font-medium hidden sm:inline">Engine Log</span>
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
              {openDropdown === 'voyage' && (
                <VoyageDropdown onFitRoute={onFitRoute} onClose={closeDropdown} />
              )}
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

            {/* Exit */}
            <button
              onClick={() => { window.close(); window.location.href = 'about:blank'; }}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
              title="Exit"
            >
              <LogOut className="w-5 h-5" />
              <span className="text-sm font-medium hidden sm:inline">Exit</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
