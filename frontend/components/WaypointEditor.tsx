'use client';

import { useCallback, useEffect, useState } from 'react';
import { Position } from '@/lib/api';

/** Wrap longitude to [-180, 180] (Leaflet can return unwrapped values). */
function wrapLng(lng: number): number {
  return ((((lng + 180) % 360) + 360) % 360) - 180;
}

interface WaypointEditorProps {
  waypoints: Position[];
  onWaypointsChange: (waypoints: Position[]) => void;
  isEditing: boolean;
  routeColor?: string;
}

/**
 * Interactive waypoint editor component.
 * Click on map to add waypoints, drag to move, right-click to delete.
 *
 * This component handles SSR by only loading react-leaflet on client side.
 */
export default function WaypointEditor(props: WaypointEditorProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return null;
  }

  // Render the actual editor only on client side
  return <WaypointEditorInner {...props} />;
}

// Inner component that uses react-leaflet
function WaypointEditorInner({
  waypoints,
  onWaypointsChange,
  isEditing,
  routeColor = '#0073e6',
}: WaypointEditorProps) {
  // Dynamic imports for react-leaflet components
  const { useMap, useMapEvents, Marker, Polyline, Popup } = require('react-leaflet');
  const L = require('leaflet');

  const map = useMap();

  // Create numbered icon
  const createNumberedIcon = useCallback((index: number, total: number) => {
    const isFirst = index === 0;
    const isLast = index === total - 1;
    const color = isFirst ? '#22c55e' : isLast ? '#ef4444' : '#0073e6';

    return L.divIcon({
      className: 'numbered-waypoint-icon',
      html: `
        <div style="
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          background-color: ${color};
          border: 2px solid white;
          border-radius: 50%;
          box-shadow: 0 2px 6px rgba(0,0,0,0.4);
          color: white;
          font-size: 11px;
          font-weight: bold;
        ">${index + 1}</div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
  }, [L]);

  // Map click handler component
  function MapClickHandler() {
    useMapEvents({
      click(e: any) {
        if (!isEditing) return;

        const newWaypoint: Position = {
          lat: e.latlng.lat,
          lon: wrapLng(e.latlng.lng),
        };

        onWaypointsChange([...waypoints, newWaypoint]);
      },
    });
    return null;
  }

  // Handle waypoint drag
  const handleDrag = useCallback(
    (index: number, e: any) => {
      const marker = e.target;
      const position = marker.getLatLng();

      const newWaypoints = [...waypoints];
      newWaypoints[index] = {
        lat: position.lat,
        lon: wrapLng(position.lng),
      };

      onWaypointsChange(newWaypoints);
    },
    [waypoints, onWaypointsChange]
  );

  // Handle waypoint deletion
  const handleDelete = useCallback(
    (index: number) => {
      const newWaypoints = waypoints.filter((_, i) => i !== index);
      onWaypointsChange(newWaypoints);
    },
    [waypoints, onWaypointsChange]
  );

  // Calculate route line positions
  const routePositions: [number, number][] = waypoints.map((wp) => [wp.lat, wp.lon]);

  return (
    <>
      <MapClickHandler />

      {/* Route polyline */}
      {routePositions.length >= 2 && (
        <Polyline
          positions={routePositions}
          pathOptions={{
            color: routeColor,
            weight: 3,
            opacity: 0.9,
          }}
        />
      )}

      {/* Waypoint markers */}
      {waypoints.map((wp, index) => (
        <Marker
          key={`wp-${index}-${wp.lat}-${wp.lon}`}
          position={[wp.lat, wp.lon]}
          icon={createNumberedIcon(index, waypoints.length)}
          draggable={isEditing && index > 0 && index < waypoints.length - 1}
          eventHandlers={{
            dragend: (e: any) => handleDrag(index, e),
            contextmenu: (e: any) => {
              e.originalEvent.preventDefault();
              if (isEditing) {
                handleDelete(index);
              }
            },
          }}
        >
          <Popup>
            <div className="text-sm">
              <div className="font-semibold">WP{index + 1}</div>
              <div className="text-gray-600">
                {wp.lat.toFixed(4)}°N, {wp.lon.toFixed(4)}°E
              </div>
              {isEditing && (
                <button
                  onClick={() => handleDelete(index)}
                  className="mt-2 px-2 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                >
                  Delete
                </button>
              )}
            </div>
          </Popup>
        </Marker>
      ))}
    </>
  );
}

/**
 * Waypoint list panel showing all waypoints with edit controls.
 */
interface WaypointListProps {
  waypoints: Position[];
  onWaypointsChange: (waypoints: Position[]) => void;
  onClear: () => void;
  totalDistance?: number;
}

export function WaypointList({
  waypoints,
  onWaypointsChange,
  onClear,
  totalDistance,
}: WaypointListProps) {
  // Calculate leg distances
  const calculateDistance = (p1: Position, p2: Position): number => {
    const R = 3440.065; // Earth radius in nm
    const lat1 = (p1.lat * Math.PI) / 180;
    const lat2 = (p2.lat * Math.PI) / 180;
    const dlat = ((p2.lat - p1.lat) * Math.PI) / 180;
    const dlon = ((p2.lon - p1.lon) * Math.PI) / 180;

    const a =
      Math.sin(dlat / 2) * Math.sin(dlat / 2) +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) * Math.sin(dlon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  };

  const moveWaypoint = (fromIndex: number, toIndex: number) => {
    const newWaypoints = [...waypoints];
    const [moved] = newWaypoints.splice(fromIndex, 1);
    newWaypoints.splice(toIndex, 0, moved);
    onWaypointsChange(newWaypoints);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-400">
          {waypoints.length} waypoint{waypoints.length !== 1 ? 's' : ''}
        </span>
        {waypoints.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs text-red-400 hover:text-red-300"
          >
            Clear All
          </button>
        )}
      </div>

      {waypoints.length === 0 ? (
        <div className="text-sm text-gray-500 italic">
          Click on the map to add waypoints
        </div>
      ) : (
        <div className="space-y-1 max-h-60 overflow-y-auto">
          {waypoints.map((wp, index) => {
            const isFirst = index === 0;
            const isLast = index === waypoints.length - 1;
            const legDist =
              index > 0 ? calculateDistance(waypoints[index - 1], wp) : null;

            return (
              <div key={index}>
                {legDist !== null && (
                  <div className="text-xs text-gray-500 pl-6 py-1">
                    ↓ {legDist.toFixed(1)} nm
                  </div>
                )}
                <div
                  className={`flex items-center justify-between p-2 rounded ${
                    isFirst
                      ? 'bg-green-500/10 border border-green-500/20'
                      : isLast
                      ? 'bg-red-500/10 border border-red-500/20'
                      : 'bg-maritime-dark'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        isFirst
                          ? 'bg-green-500 text-white'
                          : isLast
                          ? 'bg-red-500 text-white'
                          : 'bg-primary-500 text-white'
                      }`}
                    >
                      {index + 1}
                    </span>
                    <div>
                      <div className="text-xs text-gray-300">
                        {wp.lat.toFixed(3)}°, {wp.lon.toFixed(3)}°
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1">
                    {index > 0 && (
                      <button
                        onClick={() => moveWaypoint(index, index - 1)}
                        className="p-1 text-gray-400 hover:text-white"
                        title="Move up"
                      >
                        ↑
                      </button>
                    )}
                    {index < waypoints.length - 1 && (
                      <button
                        onClick={() => moveWaypoint(index, index + 1)}
                        className="p-1 text-gray-400 hover:text-white"
                        title="Move down"
                      >
                        ↓
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {totalDistance !== undefined && totalDistance > 0 && (
        <div className="pt-2 border-t border-white/10">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Total Distance:</span>
            <span className="text-white font-semibold">
              {totalDistance.toFixed(1)} nm
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
