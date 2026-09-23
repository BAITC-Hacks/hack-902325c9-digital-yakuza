import type { AgentRun, Campaign } from '../types/api';

export function explanationReport(run: AgentRun | null) {
  return run?.explanation?.explanation ?? run?.explanation;
}

export function campaignId(run: AgentRun, campaign: Campaign, index: number): string {
  return Object.entries(run.explanation?.facts ?? {}).find(([, fact]) =>
    fact.kind === 'campaign' && fact.fields.name === campaign.campaign_name)?.[0] ?? 'C' + (index + 1);
}

export function estimateLabel(value: string): string {
  return ({ pilots: 'пилоты', pilot: 'пилоты', prior: 'исторические данные', history: 'исторические данные' } as Record<string, string>)[value] ?? value;
}
