'use client';

import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from 'react-leaflet';
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

interface RouteMapProps {
  waypoints: [number, number][];
  startLabel?: string;
  endLabel?: string;
}

// Component to fit map bounds
function FitBounds({ waypoints }: { waypoints: [number, number][] }) {
  const map = useMap();

  useEffect(() => {
    if (waypoints.length > 0) {
      const bounds = L.latLngBounds(waypoints);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [waypoints, map]);

  return null;
}

export default function RouteMap({
  waypoints,
  startLabel = 'Start',
  endLabel = 'Destination',
}: RouteMapProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted || waypoints.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-maritime-dark rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading map...</p>
        </div>
      </div>
    );
  }

  const center: [number, number] = waypoints.length > 0
    ? waypoints[Math.floor(waypoints.length / 2)]
    : [45, 0];

  return (
    <div className="w-full h-full rounded-lg overflow-hidden shadow-maritime">
      <MapContainer
        center={center}
        zoom={5}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
        />
        <CountryLabels />

        {/* Route line */}
        <Polyline
          positions={waypoints}
          pathOptions={{
            color: '#3a5eae',
            weight: 3,
            opacity: 0.8,
          }}
        />

        {/* Start marker */}
        {waypoints.length > 0 && (
          <Marker position={waypoints[0]}>
            <Popup>
              <div className="text-sm font-medium">{startLabel}</div>
              <div className="text-xs text-gray-600">
                {waypoints[0][0].toFixed(4)}°N, {waypoints[0][1].toFixed(4)}°E
              </div>
            </Popup>
          </Marker>
        )}

        {/* End marker */}
        {waypoints.length > 1 && (
          <Marker position={waypoints[waypoints.length - 1]}>
            <Popup>
              <div className="text-sm font-medium">{endLabel}</div>
              <div className="text-xs text-gray-600">
                {waypoints[waypoints.length - 1][0].toFixed(4)}°N,{' '}
                {waypoints[waypoints.length - 1][1].toFixed(4)}°E
              </div>
            </Popup>
          </Marker>
        )}

        <FitBounds waypoints={waypoints} />
      </MapContainer>
    </div>
  );
}
