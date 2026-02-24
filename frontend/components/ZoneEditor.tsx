'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { FeatureGroup } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import L from 'leaflet';
import { ZoneType, ZoneInteraction, ZoneCoordinate, CreateZoneRequest } from '@/lib/api';
import { X, Save, MapPin, Shield, AlertTriangle, Ban, Info } from 'lucide-react';


interface DrawnZone {
  coordinates: ZoneCoordinate[];
  layer: L.Layer;
}

interface ZoneEditorProps {
  onSaveZone: (request: CreateZoneRequest) => Promise<void>;
  onCancel: () => void;
  isDrawing: boolean;
}

export default function ZoneEditor({ onSaveZone, onCancel, isDrawing }: ZoneEditorProps) {
  const [drawnZone, setDrawnZone] = useState<DrawnZone | null>(null);
  const [showForm, setShowForm] = useState(false);
  const featureGroupRef = useRef<L.FeatureGroup>(null);

  // Form state
  const [name, setName] = useState('');
  const [zoneType, setZoneType] = useState<ZoneType>('custom');
  const [interaction, setInteraction] = useState<ZoneInteraction>('penalty');
  const [penaltyFactor, setPenaltyFactor] = useState(1.0);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);

  // Handle polygon created
  const onCreated = useCallback((e: L.DrawEvents.Created) => {
    const layer = e.layer as L.Polygon;
    const latLngs = layer.getLatLngs()[0] as L.LatLng[];

    const coordinates: ZoneCoordinate[] = latLngs.map(ll => ({
      lat: ll.lat,
      lon: ll.lng,
    }));

    setDrawnZone({ coordinates, layer });
    setShowForm(true);
  }, []);

  // Handle delete
  const onDeleted = useCallback(() => {
    setDrawnZone(null);
    setShowForm(false);
  }, []);

  // Reset form
  const resetForm = useCallback(() => {
    setName('');
    setZoneType('custom');
    setInteraction('penalty');
    setPenaltyFactor(1.0);
    setNotes('');
    setDrawnZone(null);
    setShowForm(false);

    // Clear drawn layers
    if (featureGroupRef.current) {
      featureGroupRef.current.clearLayers();
    }
  }, []);

  // Handle save
  const handleSave = async () => {
    if (!drawnZone || !name.trim()) return;

    setSaving(true);
    try {
      await onSaveZone({
        name: name.trim(),
        zone_type: zoneType,
        interaction,
        coordinates: drawnZone.coordinates,
        penalty_factor: penaltyFactor,
        notes: notes.trim() || undefined,
      });
      resetForm();
    } catch (error) {
      console.error('Failed to save zone:', error);
      alert('Failed to save zone. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  // Handle cancel
  const handleCancel = () => {
    resetForm();
    onCancel();
  };

  if (!isDrawing) {
    return null;
  }

  return (
    <>
      {/* Drawing controls on map */}
      <FeatureGroup ref={featureGroupRef}>
        <EditControl
          position="topright"
          onCreated={onCreated}
          onDeleted={onDeleted}
          draw={{
            rectangle: false,
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false,
            polygon: {
              allowIntersection: false,
              showArea: true,
              shapeOptions: {
                color: '#f59e0b',
                fillColor: '#f59e0b',
                fillOpacity: 0.3,
              },
            },
          }}
          edit={{
            remove: true,
            edit: false,
          }}
        />
      </FeatureGroup>

      {/* Zone properties form */}
      {showForm && drawnZone && (
        <ZoneForm
          name={name}
          setName={setName}
          zoneType={zoneType}
          setZoneType={setZoneType}
          interaction={interaction}
          setInteraction={setInteraction}
          penaltyFactor={penaltyFactor}
          setPenaltyFactor={setPenaltyFactor}
          notes={notes}
          setNotes={setNotes}
          onSave={handleSave}
          onCancel={handleCancel}
          saving={saving}
          coordinateCount={drawnZone.coordinates.length}
        />
      )}
    </>
  );
}

// Zone properties form component
interface ZoneFormProps {
  name: string;
  setName: (v: string) => void;
  zoneType: ZoneType;
  setZoneType: (v: ZoneType) => void;
  interaction: ZoneInteraction;
  setInteraction: (v: ZoneInteraction) => void;
  penaltyFactor: number;
  setPenaltyFactor: (v: number) => void;
  notes: string;
  setNotes: (v: string) => void;
  onSave: () => void;
  onCancel: () => void;
  saving: boolean;
  coordinateCount: number;
}

function ZoneForm({
  name, setName,
  zoneType, setZoneType,
  interaction, setInteraction,
  penaltyFactor, setPenaltyFactor,
  notes, setNotes,
  onSave, onCancel,
  saving, coordinateCount,
}: ZoneFormProps) {
  const interactionIcons: Record<ZoneInteraction, React.ReactElement> = {
    mandatory: <MapPin className="w-4 h-4" />,
    exclusion: <Ban className="w-4 h-4" />,
    penalty: <AlertTriangle className="w-4 h-4" />,
    advisory: <Info className="w-4 h-4" />,
  };

  return (
    <div className="absolute top-4 right-16 z-[1000] bg-maritime-medium border border-white/10 rounded-lg shadow-xl p-4 w-80">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary-400" />
          New Zone
        </h3>
        <button
          onClick={onCancel}
          className="text-gray-400 hover:text-white"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-3">
        {/* Zone name */}
        <div>
          <label className="block text-sm text-gray-300 mb-1">Zone Name *</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., My Custom Zone"
            className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded text-white text-sm focus:border-primary-500 focus:outline-none"
          />
        </div>

        {/* Zone type */}
        <div>
          <label className="block text-sm text-gray-300 mb-1">Zone Type</label>
          <select
            value={zoneType}
            onChange={(e) => setZoneType(e.target.value as ZoneType)}
            className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded text-white text-sm focus:border-primary-500 focus:outline-none"
          >
            <option value="custom">Custom</option>
            <option value="eca">ECA (Emission Control)</option>
            <option value="hra">HRA (High Risk Area)</option>
            <option value="tss">TSS (Traffic Separation)</option>
            <option value="exclusion">Exclusion Zone</option>
            <option value="environmental">Environmental</option>
            <option value="ice">Ice Zone</option>
            <option value="canal">Canal/Strait</option>
          </select>
        </div>

        {/* Interaction type */}
        <div>
          <label className="block text-sm text-gray-300 mb-1">Route Interaction</label>
          <div className="grid grid-cols-2 gap-2">
            {(['mandatory', 'exclusion', 'penalty', 'advisory'] as ZoneInteraction[]).map((int) => (
              <button
                key={int}
                onClick={() => setInteraction(int)}
                className={`flex items-center gap-2 px-3 py-2 rounded text-xs font-medium transition-colors ${
                  interaction === int
                    ? int === 'exclusion'
                      ? 'bg-red-500/30 border-red-500 text-red-400 border'
                      : int === 'mandatory'
                      ? 'bg-blue-500/30 border-blue-500 text-blue-400 border'
                      : int === 'penalty'
                      ? 'bg-amber-500/30 border-amber-500 text-amber-400 border'
                      : 'bg-gray-500/30 border-gray-500 text-gray-400 border'
                    : 'bg-maritime-dark border border-white/10 text-gray-400 hover:text-white'
                }`}
              >
                {interactionIcons[int]}
                {int.charAt(0).toUpperCase() + int.slice(1)}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {interaction === 'mandatory' && 'Route must pass through this zone'}
            {interaction === 'exclusion' && 'Route cannot pass through this zone'}
            {interaction === 'penalty' && 'Route can pass but with cost penalty'}
            {interaction === 'advisory' && 'Information only, no routing impact'}
          </p>
        </div>

        {/* Penalty factor (only for penalty interaction) */}
        {interaction === 'penalty' && (
          <div>
            <label className="block text-sm text-gray-300 mb-1">
              Penalty Factor: {((penaltyFactor - 1) * 100).toFixed(0)}% extra cost
            </label>
            <input
              type="range"
              min="1"
              max="3"
              step="0.1"
              value={penaltyFactor}
              onChange={(e) => setPenaltyFactor(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span>100%</span>
              <span>200%</span>
            </div>
          </div>
        )}

        {/* Notes */}
        <div>
          <label className="block text-sm text-gray-300 mb-1">Notes (optional)</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Additional information..."
            rows={2}
            className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded text-white text-sm focus:border-primary-500 focus:outline-none resize-none"
          />
        </div>

        {/* Coordinate info */}
        <div className="text-xs text-gray-500">
          Polygon with {coordinateCount} vertices
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <button
            onClick={onCancel}
            className="flex-1 py-2 text-sm text-gray-400 bg-maritime-dark border border-white/10 rounded hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onSave}
            disabled={saving || !name.trim()}
            className="flex-1 py-2 text-sm text-white bg-primary-500 rounded hover:bg-primary-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {saving ? (
              <>Saving...</>
            ) : (
              <>
                <Save className="w-4 h-4" />
                Save Zone
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
