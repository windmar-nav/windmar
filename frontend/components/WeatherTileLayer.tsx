'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

interface WeatherTileLayerProps {
  field: string;
  forecastHour?: number;
  opacity?: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function WeatherTileLayer(props: WeatherTileLayerProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return null;

  return <WeatherTileLayerInner {...props} />;
}

function WeatherTileLayerInner({
  field,
  forecastHour = 0,
  opacity = 0.6,
}: WeatherTileLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);

  // Extend TileLayer to round fractional zoom (map uses zoomSnap=0.25)
  const RoundedTileLayer = useMemo(() => L.TileLayer.extend({
    _getZoomForUrl: function () {
      return Math.round(L.TileLayer.prototype._getZoomForUrl.call(this));
    },
  }), [L]);

  // Create the tile layer once on mount (key={field} in parent handles field changes)
  useEffect(() => {
    const PANE_NAME = 'weatherTilePane';
    if (!map.getPane(PANE_NAME)) {
      const pane = map.createPane(PANE_NAME);
      pane.style.zIndex = '300';
      pane.style.pointerEvents = 'none';
    }

    const url = `${API_BASE_URL}/api/tiles/${field}/{z}/{x}/{y}.png?h=${forecastHour}`;

    const layer = new RoundedTileLayer(url, {
      opacity,
      pane: PANE_NAME,
      tileSize: 256,
      zoomOffset: 0,
      maxNativeZoom: field === 'wind' || field === 'visibility' ? 8 : 10,
      maxZoom: 18,
      errorTileUrl: '',
      className: 'weather-tile-layer',
    });

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      map.removeLayer(layer);
      layerRef.current = null;
    };
  }, [map, L, field]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update URL when forecast hour changes (no layer re-creation needed)
  useEffect(() => {
    if (!layerRef.current) return;
    const url = `${API_BASE_URL}/api/tiles/${field}/{z}/{x}/{y}.png?h=${forecastHour}`;
    layerRef.current.setUrl(url);
  }, [forecastHour, field]);

  // Update opacity without re-creating the layer
  useEffect(() => {
    if (!layerRef.current) return;
    layerRef.current.setOpacity(opacity);
  }, [opacity]);

  return null;
}
