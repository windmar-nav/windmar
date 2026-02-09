'use client';

import { useEffect, useState, useCallback } from 'react';
import { GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import { apiClient, ZoneGeoJSON, ZoneFeature, ZoneType, ZoneInteraction, CreateZoneRequest } from '@/lib/api';

// Zone colors based on type and interaction
const ZONE_COLORS: Record<ZoneInteraction, string> = {
  mandatory: '#3b82f6',  // Blue
  exclusion: '#ef4444',  // Red
  penalty: '#f59e0b',    // Amber
  advisory: '#6b7280',   // Gray
};

const ZONE_TYPE_COLORS: Record<ZoneType, string> = {
  eca: '#22c55e',        // Green (environmental)
  seca: '#22c55e',
  hra: '#ef4444',        // Red (danger)
  tss: '#3b82f6',        // Blue (navigation)
  vts: '#3b82f6',
  exclusion: '#ef4444',
  environmental: '#22c55e',
  ice: '#06b6d4',        // Cyan
  canal: '#8b5cf6',      // Purple
  custom: '#f59e0b',     // Amber
};

interface ZoneLayerProps {
  visible?: boolean;
  visibleTypes?: string[];
  onZoneClick?: (zoneId: string, zoneName: string) => void;
}

export default function ZoneLayer({ visible = true, visibleTypes, onZoneClick }: ZoneLayerProps) {
  const [zones, setZones] = useState<ZoneGeoJSON | null>(null);
  const [loading, setLoading] = useState(true);
  const map = useMap();

  // Load zones from API
  const loadZones = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.getZones();
      setZones(data);
    } catch (error) {
      console.error('Failed to load zones:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadZones();
  }, [loadZones]);

  // Style function for GeoJSON features
  const getZoneStyle = (feature: GeoJSON.Feature | undefined) => {
    if (!feature?.properties) {
      return {
        fillColor: '#6b7280',
        fillOpacity: 0.2,
        color: '#6b7280',
        weight: 2,
      };
    }

    const props = feature.properties as ZoneFeature['properties'];
    const interaction = props.interaction as ZoneInteraction;
    const zoneType = props.zone_type as ZoneType;

    // Use interaction color for stroke, type color for fill
    const strokeColor = ZONE_COLORS[interaction] || '#6b7280';
    const fillColor = ZONE_TYPE_COLORS[zoneType] || ZONE_COLORS[interaction] || '#6b7280';

    // Adjust opacity based on interaction
    let fillOpacity = 0.15;
    let weight = 2;
    let dashArray: string | undefined;

    switch (interaction) {
      case 'exclusion':
        fillOpacity = 0.3;
        weight = 3;
        break;
      case 'mandatory':
        fillOpacity = 0.2;
        weight = 2;
        dashArray = '5, 5';
        break;
      case 'penalty':
        fillOpacity = 0.15;
        weight = 2;
        break;
      case 'advisory':
        fillOpacity = 0.1;
        weight: 1;
        dashArray = '2, 4';
        break;
    }

    return {
      fillColor,
      fillOpacity,
      color: strokeColor,
      weight,
      dashArray,
    };
  };

  // Popup content for zones
  const onEachFeature = (feature: GeoJSON.Feature, layer: L.Layer) => {
    if (!feature.properties) return;

    const props = feature.properties as ZoneFeature['properties'];
    const interactionLabel = {
      mandatory: 'Mandatory Transit',
      exclusion: 'Exclusion Zone',
      penalty: 'Cost Penalty',
      advisory: 'Advisory Only',
    }[props.interaction] || props.interaction;

    const penaltyText = props.penalty_factor > 1
      ? `+${((props.penalty_factor - 1) * 100).toFixed(0)}% cost`
      : 'No penalty';

    const popupContent = `
      <div style="min-width: 180px;">
        <strong style="font-size: 14px;">${props.name}</strong>
        <hr style="margin: 4px 0; border-color: #374151;" />
        <div style="font-size: 12px; color: #9ca3af;">
          <div><strong>Type:</strong> ${props.zone_type.toUpperCase()}</div>
          <div><strong>Interaction:</strong> ${interactionLabel}</div>
          <div><strong>Penalty:</strong> ${penaltyText}</div>
          ${props.notes ? `<div style="margin-top: 4px; font-style: italic;">${props.notes}</div>` : ''}
          ${props.is_builtin ? '<div style="margin-top: 4px; color: #6b7280;">Built-in zone</div>' : '<div style="margin-top: 4px; color: #f59e0b;">Custom zone</div>'}
        </div>
      </div>
    `;

    layer.bindPopup(popupContent, {
      className: 'zone-popup',
    });

    // Handle click for external handlers
    if (onZoneClick && feature.id) {
      layer.on('click', () => {
        onZoneClick(String(feature.id), props.name);
      });
    }
  };

  if (!visible || !zones) {
    return null;
  }

  // Filter features by visible zone types if provided
  const filteredData = visibleTypes && visibleTypes.length > 0
    ? {
        ...zones,
        features: zones.features.filter(
          (f) => visibleTypes.includes(f.properties.zone_type)
        ),
      }
    : zones;

  if (filteredData.features.length === 0) {
    return null;
  }

  return (
    <GeoJSON
      key={JSON.stringify(filteredData)} // Force re-render when zones or filter changes
      data={filteredData as GeoJSON.GeoJsonObject}
      style={getZoneStyle}
      onEachFeature={onEachFeature}
    />
  );
}

// Hook for zone management
export function useZones() {
  const [zones, setZones] = useState<ZoneGeoJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadZones = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getZones();
      setZones(data);
    } catch (err) {
      setError('Failed to load zones');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const createZone = useCallback(async (request: CreateZoneRequest) => {
    try {
      setLoading(true);
      setError(null);
      await apiClient.createZone(request);
      await loadZones(); // Reload zones after creating
    } catch (err) {
      setError('Failed to create zone');
      console.error(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadZones]);

  const deleteZone = useCallback(async (zoneId: string) => {
    try {
      setLoading(true);
      setError(null);
      await apiClient.deleteZone(zoneId);
      await loadZones(); // Reload zones after deleting
    } catch (err) {
      setError('Failed to delete zone');
      console.error(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadZones]);

  useEffect(() => {
    loadZones();
  }, [loadZones]);

  return {
    zones,
    loading,
    error,
    loadZones,
    createZone,
    deleteZone,
  };
}
