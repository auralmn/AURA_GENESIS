export interface FiringEvent {
  timestamp: number;
  neuron_id: string;
  region: string;
  firing_strength: number;
  trigger_source: string;
  trigger_details: string;
  context: string;
}

export interface NeuronStatus {
  neuron_id: string;
  region: string;
  firing_rate: number;
  activity_state: string;
  weight_magnitude: number;
  last_fire_time: number;
  health_status: 'healthy' | 'warning' | 'critical';
  firing_history: FiringEvent[];
}

export interface SystemAlert {
  id: string;
  timestamp: number;
  severity: 'critical' | 'warning' | 'info';
  component: string;
  message: string;
  details?: string;
}

export interface HealthSnapshot {
  timestamp: number;
  hormone_levels: Record<string, number>;
  endocrine_effects: Record<string, number>;
  router_usage: Record<string, number>;
  expert_utilization: Record<string, number>;
  energy_consumption: number;
  prediction_accuracy: number;
  system_metrics: Record<string, number>;
  routing_decisions: Array<{
    expert: string;
    confidence: number;
    gate: number;
  }>;
  neuron_statuses: NeuronStatus[];
  active_alerts: SystemAlert[];
}

export interface FiringPatterns {
  total_firings: number;
  firing_rates: Record<string, number>;
  trigger_patterns: Record<string, number>;
  most_active_neurons: Array<{
    neuron_id: string;
    firing_count: number;
  }>;
  region_summaries: Record<string, {
    total_firings: number;
    average_firing_strength: number;
  }>;
}

export interface WebSocketMessage {
  type: 'health_snapshot' | 'firing_event' | 'alert' | 'system_status';
  data: HealthSnapshot | FiringEvent | SystemAlert | any;
  timestamp: number;
}
