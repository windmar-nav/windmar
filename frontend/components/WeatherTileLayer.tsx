'use client';

import { useEffect, useRef, useState } from 'react';

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
  const prevUrlRef = useRef<string>('');

  useEffect(() => {
    const url = `${API_BASE_URL}/api/tiles/${field}/{z}/{x}/{y}.png?h=${forecastHour}`;

    if (layerRef.current) {
      // Field or forecast hour changed — swap URL (Leaflet handles tile refresh)
      if (prevUrlRef.current !== url) {
        layerRef.current.setUrl(url);
        prevUrlRef.current = url;
      }
      layerRef.current.setOpacity(opacity);
      return;
    }

    const layer = L.tileLayer(url, {
      opacity,
      tileSize: 256,
      zoomOffset: 0,
      maxNativeZoom: field === 'wind' || field === 'visibility' ? 8 : 10,
      maxZoom: 18,
      errorTileUrl: '',
      className: 'weather-tile-layer',
    });

    layer.addTo(map);
    layerRef.current = layer;
    prevUrlRef.current = url;

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
        prevUrlRef.current = '';
      }
    };
  }, [field, forecastHour, opacity, map, L]);

  return null;
}
