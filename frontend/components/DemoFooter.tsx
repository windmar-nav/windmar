'use client';

import { DEMO_MODE } from '@/lib/demoMode';

export function DemoFooter() {
  if (!DEMO_MODE) return null;

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 bg-gray-900/90 backdrop-blur border-t border-white/10 px-4 py-1.5 text-[10px] text-gray-500 text-center">
      Weather data: <a href="https://marine.copernicus.eu" target="_blank" rel="noopener noreferrer" className="underline hover:text-gray-300">Copernicus Marine Service</a>
      {' '}&bull;{' '}
      <a href="https://www.ncep.noaa.gov/products/weather/marine/" target="_blank" rel="noopener noreferrer" className="underline hover:text-gray-300">NOAA/NCEP GFS</a>
      {' '}&mdash;{' '}
      Pre-loaded snapshot &bull; No live API calls
    </footer>
  );
}
