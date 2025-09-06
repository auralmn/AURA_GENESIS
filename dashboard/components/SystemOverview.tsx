'use client';

import { HealthSnapshot } from '@/types/aura';
import { Activity, Brain, Zap, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

interface SystemOverviewProps {
  snapshot: HealthSnapshot;
}

export default function SystemOverview({ snapshot }: SystemOverviewProps) {
  const { 
    hormone_levels, 
    expert_utilization, 
    energy_consumption, 
    prediction_accuracy,
    active_alerts,
    neuron_statuses 
  } = snapshot;

  const healthyNeurons = neuron_statuses.filter(n => n.health_status === 'healthy').length;
  const warningNeurons = neuron_statuses.filter(n => n.health_status === 'warning').length;
  const criticalNeurons = neuron_statuses.filter(n => n.health_status === 'critical').length;

  const criticalAlerts = active_alerts.filter(a => a.severity === 'critical').length;
  const warningAlerts = active_alerts.filter(a => a.severity === 'warning').length;

  const avgExpertUtilization = Object.values(expert_utilization).reduce((a, b) => a + b, 0) / Object.keys(expert_utilization).length;

  const metrics = [
    {
      title: 'Neuron Health',
      value: `${healthyNeurons}/${neuron_statuses.length}`,
      subtitle: `${((healthyNeurons / neuron_statuses.length) * 100).toFixed(1)}% healthy`,
      icon: <Brain className="w-6 h-6" />,
      color: healthyNeurons === neuron_statuses.length ? 'text-green-400' : 'text-yellow-400',
      bgColor: healthyNeurons === neuron_statuses.length ? 'bg-green-900/20' : 'bg-yellow-900/20',
    },
    {
      title: 'Active Alerts',
      value: active_alerts.length.toString(),
      subtitle: `${criticalAlerts} critical, ${warningAlerts} warning`,
      icon: <AlertTriangle className="w-6 h-6" />,
      color: criticalAlerts > 0 ? 'text-red-400' : warningAlerts > 0 ? 'text-yellow-400' : 'text-green-400',
      bgColor: criticalAlerts > 0 ? 'bg-red-900/20' : warningAlerts > 0 ? 'bg-yellow-900/20' : 'bg-green-900/20',
    },
    {
      title: 'Energy Consumption',
      value: `${(energy_consumption * 1e12).toFixed(2)} pJ`,
      subtitle: 'Per operation',
      icon: <Zap className="w-6 h-6" />,
      color: 'text-blue-400',
      bgColor: 'bg-blue-900/20',
    },
    {
      title: 'Prediction Accuracy',
      value: `${(prediction_accuracy * 100).toFixed(1)}%`,
      subtitle: 'Current accuracy',
      icon: <TrendingUp className="w-6 h-6" />,
      color: prediction_accuracy > 0.8 ? 'text-green-400' : prediction_accuracy > 0.6 ? 'text-yellow-400' : 'text-red-400',
      bgColor: prediction_accuracy > 0.8 ? 'bg-green-900/20' : prediction_accuracy > 0.6 ? 'bg-yellow-900/20' : 'bg-red-900/20',
    },
    {
      title: 'Expert Utilization',
      value: `${(avgExpertUtilization * 100).toFixed(1)}%`,
      subtitle: 'Average across experts',
      icon: <Activity className="w-6 h-6" />,
      color: avgExpertUtilization > 0.5 ? 'text-green-400' : avgExpertUtilization > 0.2 ? 'text-yellow-400' : 'text-red-400',
      bgColor: avgExpertUtilization > 0.5 ? 'bg-green-900/20' : avgExpertUtilization > 0.2 ? 'bg-yellow-900/20' : 'bg-red-900/20',
    },
    {
      title: 'System Status',
      value: criticalAlerts === 0 ? 'Healthy' : 'Issues',
      subtitle: criticalAlerts === 0 ? 'All systems operational' : 'Attention required',
      icon: criticalAlerts === 0 ? <CheckCircle className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />,
      color: criticalAlerts === 0 ? 'text-green-400' : 'text-red-400',
      bgColor: criticalAlerts === 0 ? 'bg-green-900/20' : 'bg-red-900/20',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
      {metrics.map((metric, index) => (
        <div
          key={index}
          className={`metric-card ${metric.bgColor} border-l-4 border-l-current ${metric.color} ${criticalAlerts > 0 ? 'pulse-glow' : ''}`}
        >
          <div className="flex items-center justify-between mb-2">
            <div className={`${metric.color}`}>
              {metric.icon}
            </div>
            <div className="text-right">
              <div className={`text-2xl font-bold ${metric.color}`}>
                {metric.value}
              </div>
              <div className="text-sm opacity-75">
                {metric.subtitle}
              </div>
            </div>
          </div>
          <div className="text-lg font-semibold">
            {metric.title}
          </div>
        </div>
      ))}
    </div>
  );
}
