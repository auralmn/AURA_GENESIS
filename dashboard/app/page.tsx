'use client';

import { useState, useEffect } from 'react';
import { HealthSnapshot, FiringEvent, SystemAlert } from '@/types/aura';
import { auraWebSocket } from '@/lib/websocket';
import SystemOverview from '@/components/SystemOverview';
import AlertPanel from '@/components/AlertPanel';
import NeuronGrid from '@/components/NeuronGrid';
import MetricsChart from '@/components/MetricsChart';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentSnapshot, setCurrentSnapshot] = useState<HealthSnapshot | null>(null);
  const [firingEvents, setFiringEvents] = useState<FiringEvent[]>([]);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Connect to WebSocket
    const socket = auraWebSocket.connect();
    
    // Subscribe to health updates
    auraWebSocket.subscribeToHealthUpdates((data: HealthSnapshot) => {
      setCurrentSnapshot(data);
      setAlerts(data.active_alerts);
      setIsLoading(false);
    });

    // Subscribe to firing events
    auraWebSocket.subscribeToFiringEvents((data: FiringEvent) => {
      setFiringEvents(prev => [data, ...prev].slice(0, 100)); // Keep last 100 events
    });

    // Subscribe to alerts
    auraWebSocket.subscribeToAlerts((data: SystemAlert) => {
      setAlerts(prev => [data, ...prev].slice(0, 50)); // Keep last 50 alerts
    });

    // Check connection status
    const checkConnection = () => {
      setIsConnected(auraWebSocket.isConnected());
    };

    const interval = setInterval(checkConnection, 1000);
    checkConnection();

    return () => {
      clearInterval(interval);
      auraWebSocket.disconnect();
    };
  }, []);

  const handleReconnect = () => {
    auraWebSocket.disconnect();
    setTimeout(() => {
      auraWebSocket.connect();
    }, 1000);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Connecting to AURA...</h2>
          <p className="text-gray-400">Establishing WebSocket connection</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            AURA Health Dashboard
          </h1>
          <p className="text-gray-400 mt-2">
            Real-time monitoring of neural network health and performance
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
            isConnected 
              ? 'bg-green-900/20 text-green-400 border border-green-500' 
              : 'bg-red-900/20 text-red-400 border border-red-500'
          }`}>
            {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          
          {!isConnected && (
            <button
              onClick={handleReconnect}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Reconnect
            </button>
          )}
        </div>
      </div>

      {/* System Overview */}
      {currentSnapshot && <SystemOverview snapshot={currentSnapshot} />}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Alerts Panel */}
        <AlertPanel alerts={alerts} />
        
        {/* Neuron Grid */}
        {currentSnapshot && (
          <NeuronGrid neurons={currentSnapshot.neuron_statuses} />
        )}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Hormone Levels Chart */}
        {currentSnapshot && (
          <MetricsChart
            data={Object.entries(currentSnapshot.hormone_levels).map(([key, value]) => ({
              name: key.replace('HormoneType.', ''),
              value: value,
              timestamp: currentSnapshot.timestamp
            }))}
            type="bar"
            dataKey="value"
            title="Hormone Levels"
            color="#8b5cf6"
          />
        )}

        {/* Expert Utilization Chart */}
        {currentSnapshot && (
          <MetricsChart
            data={Object.entries(currentSnapshot.expert_utilization).map(([key, value]) => ({
              name: key,
              value: value,
              timestamp: currentSnapshot.timestamp
            }))}
            type="pie"
            dataKey="value"
            title="Expert Utilization"
          />
        )}
      </div>

      {/* Firing Events Timeline */}
      {firingEvents.length > 0 && (
        <div className="metric-card">
          <h3 className="text-lg font-semibold mb-4">Recent Firing Events</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {firingEvents.slice(0, 20).map((event, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-gray-700"
              >
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                  <div>
                    <div className="font-mono text-sm">
                      {event.neuron_id}
                    </div>
                    <div className="text-xs text-gray-400">
                      {event.region} • {event.trigger_source}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-mono">
                    {event.firing_strength.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-400">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 text-center text-gray-400 text-sm">
        <p>AURA Neural Network Health Monitoring System</p>
        <p>Last updated: {currentSnapshot ? new Date(currentSnapshot.timestamp).toLocaleString() : 'Never'}</p>
      </div>
    </div>
  );
}
