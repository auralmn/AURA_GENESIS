'use client';

import { NeuronStatus } from '@/types/aura';
import { Activity, Brain, Zap } from 'lucide-react';
import { useState } from 'react';

interface NeuronGridProps {
  neurons: NeuronStatus[];
  maxNeurons?: number;
}

export default function NeuronGrid({ neurons, maxNeurons = 50 }: NeuronGridProps) {
  const [selectedRegion, setSelectedRegion] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'firing_rate' | 'weight_magnitude' | 'last_fire_time'>('firing_rate');

  const regions = ['all', ...Array.from(new Set(neurons.map(n => n.region)))];
  
  const filteredNeurons = neurons
    .filter(neuron => selectedRegion === 'all' || neuron.region === selectedRegion)
    .sort((a, b) => {
      switch (sortBy) {
        case 'firing_rate':
          return b.firing_rate - a.firing_rate;
        case 'weight_magnitude':
          return b.weight_magnitude - a.weight_magnitude;
        case 'last_fire_time':
          return b.last_fire_time - a.last_fire_time;
        default:
          return 0;
      }
    })
    .slice(0, maxNeurons);

  const getHealthClass = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'neuron-healthy';
      case 'warning':
        return 'neuron-warning';
      case 'critical':
        return 'neuron-critical';
      default:
        return 'text-gray-400';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <Activity className="w-3 h-3 text-green-400" />;
      case 'warning':
        return <Zap className="w-3 h-3 text-yellow-400" />;
      case 'critical':
        return <Brain className="w-3 h-3 text-red-400" />;
      default:
        return <Activity className="w-3 h-3 text-gray-400" />;
    }
  };

  return (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          Neuron Status ({filteredNeurons.length})
        </h3>
        <div className="flex gap-2">
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
          >
            {regions.map(region => (
              <option key={region} value={region}>
                {region === 'all' ? 'All Regions' : region}
              </option>
            ))}
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
          >
            <option value="firing_rate">Firing Rate</option>
            <option value="weight_magnitude">Weight Magnitude</option>
            <option value="last_fire_time">Last Fire Time</option>
          </select>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
        {filteredNeurons.map((neuron) => (
          <div
            key={neuron.neuron_id}
            className={`p-3 rounded-lg border border-gray-600 bg-gray-800/30 ${getHealthClass(neuron.health_status)}`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getHealthIcon(neuron.health_status)}
                <span className="font-mono text-sm font-medium">
                  {neuron.neuron_id}
                </span>
              </div>
              <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                {neuron.region}
              </span>
            </div>
            
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span>Firing Rate:</span>
                <span className="font-mono">
                  {neuron.firing_rate.toFixed(3)} Hz
                </span>
              </div>
              <div className="flex justify-between">
                <span>Weight:</span>
                <span className="font-mono">
                  {neuron.weight_magnitude.toExponential(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Activity:</span>
                <span className="capitalize">
                  {neuron.activity_state}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Last Fire:</span>
                <span className="font-mono">
                  {neuron.last_fire_time > 0 
                    ? `${(Date.now() - neuron.last_fire_time * 1000).toFixed(0)}ms ago`
                    : 'Never'
                  }
                </span>
              </div>
              {neuron.firing_history.length > 0 && (
                <div className="flex justify-between">
                  <span>Recent Fires:</span>
                  <span className="font-mono">
                    {neuron.firing_history.length}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
