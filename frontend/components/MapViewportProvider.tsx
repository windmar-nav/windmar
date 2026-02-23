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
    const margin = 0.35;

    onViewportChange({
      bounds: {
        lat_min: Math.max(-85, b.getSouth() - latSpan * margin),
        lat_max: Math.min(85, b.getNorth() + latSpan * margin),
        lon_min: Math.max(-180, b.getWest() - lngSpan * margin),
        lon_max: Math.min(180, b.getEast() + lngSpan * margin),
      },
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
