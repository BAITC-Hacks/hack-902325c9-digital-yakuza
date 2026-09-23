export type RunStatus = 'running' | 'completed' | 'failed';
export type DashboardStatus = 'idle' | 'running' | 'completed' | 'error';

export interface Segment {
  [key: string]: string | number | null;
  customers: number;
  mean_predicted_arpu: number;
}

export interface Tariff {
  tariff_plan_code: string;
  price_tariff: number;
  Data_in_PKG: number;
  Min_another_operator_in_PKG: number;
  Min_another_operator_and_city_in_PKG: number;
  description: string;
}

export interface CaseSummary {
  case: string;
  environment: string;
  synthetic_data: boolean;
  customers: number;
  baseline_total_arpu: number;
  mean_predicted_arpu: number;
  tariffs: Tariff[];
  segments: Record<string, Segment[]>;
  channels: Record<string, { cost_per_contact: number; conversion_multiplier: number }>;
  constraints: {
    max_campaigns: number;
    max_campaign_size: number;
    max_contacts: number;
    budget: number;
    max_pilots: number;
    min_pilot_size: number;
    max_pilot_size: number;
    max_runtime_seconds: number;
    pilots_consume_budget_and_contacts: boolean;
  };
}

export interface Campaign {
  [key: string]: unknown;
  campaign_name: string;
  target_tariff: string;
  channel: string;
  n_contacts?: number;
  cost?: number;
  gross_lift?: number;
}

export interface Pilot {
  sequence: number;
  hypothesis: string;
  segment: Record<string, unknown>;
  target_tariff: string;
  channel: string;
  pilot_size: number;
  observed_lift: number | null;
  cost: number;
}

export interface AgentRun {
  run_id: string;
  status: RunStatus;
  seed: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  environment: string;
  metrics: Record<string, unknown> | null;
  campaigns: Campaign[];
}

export interface PilotsResponse {
  run_id: string;
  status: RunStatus;
  pilots: Pilot[];
}
