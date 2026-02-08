'use client';

import { Navigation, BarChart3 } from 'lucide-react';

export type ActiveTab = 'route' | 'analysis';

interface TabPanelProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  children: React.ReactNode;
}

export default function TabPanel({ activeTab, onTabChange, children }: TabPanelProps) {
  return (
    <div className="flex flex-col min-h-0 h-full">
      {/* Tab buttons */}
      <div className="flex border-b border-white/10 shrink-0">
        <TabButton
          icon={<Navigation className="w-4 h-4" />}
          label="Route"
          active={activeTab === 'route'}
          onClick={() => onTabChange('route')}
        />
        <TabButton
          icon={<BarChart3 className="w-4 h-4" />}
          label="Analysis"
          active={activeTab === 'analysis'}
          onClick={() => onTabChange('analysis')}
        />
      </div>

      {/* Tab content — scrollable */}
      <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3">
        {children}
      </div>
    </div>
  );
}

function TabButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
        active
          ? 'text-primary-400 border-b-2 border-primary-400 bg-primary-500/5'
          : 'text-gray-400 hover:text-white border-b-2 border-transparent'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}
