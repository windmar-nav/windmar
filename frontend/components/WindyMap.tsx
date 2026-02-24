'use client';

import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline, useMap } from 'react-leaflet';
import CountryLabels from '@/components/CountryLabels';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet icon issue in Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface WindyMapProps {
  latitude: number;
  longitude: number;
  heading?: number;
  sog?: number;
  zoom?: number;
  showWindy?: boolean;
  trackPoints?: [number, number][];
  className?: string;
}

// Custom vessel icon
const createVesselIcon = (heading: number) => {
  return L.divIcon({
    className: 'vessel-icon',
    html: `
      <div style="transform: rotate(${heading}deg); width: 24px; height: 24px;">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L6 20L12 16L18 20L12 2Z" fill="#22d3ee" stroke="#0891b2" stroke-width="1"/>
        </svg>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
};

// Component to follow vessel position
function FollowVessel({ position }: { position: [number, number] }) {
  const map = useMap();

  useEffect(() => {
    map.setView(position, map.getZoom(), { animate: true });
  }, [position, map]);

  return null;
}

export default function WindyMap({
  latitude,
  longitude,
  heading = 0,
  sog = 0,
  zoom = 8,
  showWindy = true,
  trackPoints = [],
  className = '',
}: WindyMapProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [followVessel, setFollowVessel] = useState(true);
  const [windyLayer, setWindyLayer] = useState<'wind' | 'waves' | 'currents' | 'pressure' | 'temp'>('wind');

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return (
      <div className={`flex items-center justify-center bg-maritime-dark rounded-lg ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400 mx-auto mb-2" />
          <p className="text-gray-400 text-sm">Loading map...</p>
        </div>
      </div>
    );
  }

  const position: [number, number] = [latitude, longitude];

  // Windy tile layers
  const windyTileUrls: Record<string, string> = {
    wind: 'https://tiles.windy.com/tiles/v10.0/wind/{z}/{x}/{y}.png',
    waves: 'https://tiles.windy.com/tiles/v10.0/waves/{z}/{x}/{y}.png',
    currents: 'https://tiles.windy.com/tiles/v10.0/currents/{z}/{x}/{y}.png',
    pressure: 'https://tiles.windy.com/tiles/v10.0/pressure/{z}/{x}/{y}.png',
    temp: 'https://tiles.windy.com/tiles/v10.0/temp/{z}/{x}/{y}.png',
  };

  return (
    <div className={`bg-maritime-dark/80 backdrop-blur-sm rounded-lg border border-white/10 overflow-hidden ${className}`}>
      {/* Header with controls */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/10">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-white">Position</span>
          <span className="text-xs text-gray-400">
            {latitude.toFixed(4)}°, {longitude.toFixed(4)}°
          </span>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setFollowVessel(!followVessel)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              followVessel
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-dark text-gray-400 hover:text-white'
            }`}
          >
            Follow
          </button>
        </div>
      </div>

      {/* Layer selection (like MIROS right sidebar) */}
      <div className="absolute right-2 top-14 z-[1000] flex flex-col space-y-1">
        {[
          { key: 'wind', label: 'Wind', icon: '💨' },
          { key: 'waves', label: 'Waves', icon: '🌊' },
          { key: 'currents', label: 'Currents', icon: '🔄' },
          { key: 'pressure', label: 'Pressure', icon: '📊' },
          { key: 'temp', label: 'Temp', icon: '🌡️' },
        ].map((layer) => (
          <button
            key={layer.key}
            onClick={() => setWindyLayer(layer.key as any)}
            className={`w-8 h-8 rounded flex items-center justify-center text-sm transition-colors ${
              windyLayer === layer.key
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-dark/80 text-gray-400 hover:text-white border border-white/10'
            }`}
            title={layer.label}
          >
            {layer.icon}
          </button>
        ))}
      </div>

      {/* Map */}
      <div className="h-[400px]">
        <MapContainer
          center={position}
          zoom={zoom}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
        >
          {/* Base map - dark theme (no labels, country names added via overlay) */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
          />
          <CountryLabels />

          {/* Windy overlay - using iframe approach for real Windy */}
          {showWindy && (
            <TileLayer
              url={`https://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?appid=demo`}
              opacity={0.6}
            />
          )}

          {/* Track line */}
          {trackPoints.length > 1 && (
            <Polyline
              positions={trackPoints}
              pathOptions={{
                color: '#22d3ee',
                weight: 2,
                opacity: 0.6,
                dashArray: '5, 5',
              }}
            />
          )}

          {/* Vessel marker */}
          <Marker position={position} icon={createVesselIcon(heading)}>
            <Popup>
              <div className="text-sm">
                <div className="font-medium">Vessel Position</div>
                <div className="text-gray-600">
                  Lat: {latitude.toFixed(4)}°<br />
                  Lon: {longitude.toFixed(4)}°<br />
                  HDG: {heading.toFixed(1)}°<br />
                  SOG: {sog.toFixed(1)} kts
                </div>
              </div>
            </Popup>
          </Marker>

          {/* Position accuracy circle */}
          <Circle
            center={position}
            radius={100}
            pathOptions={{
              color: '#22d3ee',
              fillColor: '#22d3ee',
              fillOpacity: 0.1,
              weight: 1,
            }}
          />

          {followVessel && <FollowVessel position={position} />}
        </MapContainer>
      </div>

      {/* Windy attribution */}
      <div className="px-4 py-2 border-t border-white/10 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">Weather data:</span>
          <a
            href="https://www.windy.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary-400 hover:underline"
          >
            Windy.com
          </a>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            CMEMS
          </span>
        </div>
      </div>
    </div>
  );
}
