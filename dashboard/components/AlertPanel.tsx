'use client';

import { SystemAlert } from '@/types/aura';
import { AlertTriangle, Info, XCircle } from 'lucide-react';
import { useState } from 'react';

interface AlertPanelProps {
  alerts: SystemAlert[];
  maxAlerts?: number;
}

export default function AlertPanel({ alerts, maxAlerts = 10 }: AlertPanelProps) {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  const handleDismiss = (alertId: string) => {
    setDismissedAlerts(prev => new Set([...prev, alertId]));
  };

  const visibleAlerts = alerts
    .filter(alert => !dismissedAlerts.has(alert.id))
    .slice(0, maxAlerts);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case 'info':
        return <Info className="w-4 h-4 text-blue-400" />;
      default:
        return <Info className="w-4 h-4 text-gray-400" />;
    }
  };

  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'alert-critical';
      case 'warning':
        return 'alert-warning';
      case 'info':
        return 'alert-info';
      default:
        return 'bg-gray-800/50 border-gray-700 text-gray-300';
    }
  };

  if (visibleAlerts.length === 0) {
    return (
      <div className="metric-card">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-green-400" />
          System Alerts
        </h3>
        <div className="text-green-400 text-center py-4">
          ✅ All systems healthy - no active alerts
        </div>
      </div>
    );
  }

  return (
    <div className="metric-card">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-red-400" />
        System Alerts ({visibleAlerts.length})
      </h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {visibleAlerts.map((alert) => (
          <div
            key={alert.id}
            className={`p-3 rounded-lg border ${getSeverityClass(alert.severity)} flex items-start justify-between gap-3`}
          >
            <div className="flex items-start gap-2 flex-1">
              {getSeverityIcon(alert.severity)}
              <div className="flex-1">
                <div className="font-medium text-sm">
                  {alert.component.toUpperCase()}
                </div>
                <div className="text-sm opacity-90">
                  {alert.message}
                </div>
                {alert.details && (
                  <div className="text-xs opacity-75 mt-1">
                    {alert.details}
                  </div>
                )}
                <div className="text-xs opacity-60 mt-1">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
            <button
              onClick={() => handleDismiss(alert.id)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <XCircle className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
