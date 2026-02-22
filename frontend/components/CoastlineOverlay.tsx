'use client';

/**
 * CoastlineOverlay — renders GSHHS vector land polygons above weather layers.
 *
 * Fetches simplified coastline GeoJSON from /api/coastline for the current
 * viewport and renders it as a dark fill polygon, masking any weather grid
 * artifacts that leak onto land areas.
 */

import { useEffect, useState, useRef } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';

const LAND_FILL = '#0a0f1a';
const LAND_STROKE = '#0a0f1a';
const DEBOUNCE_MS = 500;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function CoastlineOverlay() {
  const map = useMap();
  const layerRef = useRef<L.GeoJSON | null>(null);
  const [geojson, setGeojson] = useState<any>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Create a custom pane above tilePane (200) but below overlayPane (400)
  useEffect(() => {
    if (!map.getPane('coastlinePane')) {
      const pane = map.createPane('coastlinePane');
      pane.style.zIndex = '350';
      pane.style.pointerEvents = 'none';
    }
  }, [map]);

  // Fetch coastline for current viewport (debounced)
  useEffect(() => {
    const fetchCoastline = () => {
      const bounds = map.getBounds();
      const pad = 2; // degrees padding
      const params = new URLSearchParams({
        lat_min: String(Math.max(-85, bounds.getSouth() - pad)),
        lat_max: String(Math.min(85, bounds.getNorth() + pad)),
        lon_min: String(Math.max(-180, bounds.getWest() - pad)),
        lon_max: String(Math.min(180, bounds.getEast() + pad)),
        simplify: String(map.getZoom() >= 8 ? 0.002 : map.getZoom() >= 5 ? 0.005 : 0.02),
      });

      const key = localStorage.getItem('windmar_api_key');
      const headers: Record<string, string> = {};
      if (key) headers['X-API-Key'] = key;

      fetch(`${API_BASE}/api/coastline?${params}`, { headers })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data) setGeojson(data);
        })
        .catch(() => {});
    };

    const onMove = () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(fetchCoastline, DEBOUNCE_MS);
    };

    // Initial fetch
    fetchCoastline();

    map.on('moveend', onMove);
    map.on('zoomend', onMove);

    return () => {
      map.off('moveend', onMove);
      map.off('zoomend', onMove);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [map]);

  // Render / update the GeoJSON layer
  useEffect(() => {
    if (!geojson) return;

    // Remove previous layer
    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    const layer = L.geoJSON(geojson, {
      pane: 'coastlinePane',
      style: {
        fillColor: LAND_FILL,
        fillOpacity: 1,
        color: LAND_STROKE,
        weight: 0.5,
        opacity: 1,
      },
      interactive: false,
    });

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [geojson, map]);

  return null;
}
