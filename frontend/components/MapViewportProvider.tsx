'use client';

import { useRef, useEffect, useCallback } from 'react';
import { useMapEvents, useMap } from 'react-leaflet';

export interface ViewportInfo {
  bounds: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  zoom: number;
}

interface MapViewportProviderProps {
  onViewportChange: (viewport: ViewportInfo) => void;
}

export default function MapViewportProvider({ onViewportChange }: MapViewportProviderProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const map = useMap();

  const emitViewport = useCallback(() => {
    const b = map.getBounds();
    const latSpan = b.getNorth() - b.getSouth();
    const lngSpan = b.getEast() - b.getWest();
    const margin = 0.5;

    // Expand by margin, then cap to backend limits (45° lat × 90° lon)
    const MAX_LAT_SPAN = 45;
    const MAX_LON_SPAN = 90;
    let lat_min = Math.max(-85, b.getSouth() - latSpan * margin);
    let lat_max = Math.min(85, b.getNorth() + latSpan * margin);
    let lon_min = Math.max(-180, b.getWest() - lngSpan * margin);
    let lon_max = Math.min(180, b.getEast() + lngSpan * margin);
    if (lat_max - lat_min > MAX_LAT_SPAN) {
      const mid = (lat_min + lat_max) / 2;
      lat_min = mid - MAX_LAT_SPAN / 2;
      lat_max = mid + MAX_LAT_SPAN / 2;
    }
    if (lon_max - lon_min > MAX_LON_SPAN) {
      const mid = (lon_min + lon_max) / 2;
      lon_min = mid - MAX_LON_SPAN / 2;
      lon_max = mid + MAX_LON_SPAN / 2;
    }

    onViewportChange({
      bounds: { lat_min, lat_max, lon_min, lon_max },
      zoom: map.getZoom(),
    });
  }, [map, onViewportChange]);

  const debouncedEmit = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(emitViewport, 600);
  }, [emitViewport]);

  useMapEvents({
    moveend: debouncedEmit,
    zoomend: debouncedEmit,
  });

  // Fire once on mount for initial viewport
  useEffect(() => {
    emitViewport();
  }, [emitViewport]);

  return null;
}
