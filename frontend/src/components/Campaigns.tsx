import type { AgentRun } from '../types/api';
import { DataTable, Notice, number, SegmentDetails, Stat } from './Ui';

export function Campaigns({ run }: { run: AgentRun | null }) {
  if (!run || run.status !== 'completed') return <Notice>{run?.status === 'running'
    ? 'Агент выполняет пилоты. Финальные кампании появятся после завершения.'
    : 'Завершите запуск агента, чтобы получить финальные кампании.'}</Notice>;
  return <>
    {run.metrics && <div className="stats">
      <Stat label="Финальные кампании" value={number(run.campaigns.length)} />
      <Stat label="Всего контактов" value={number(run.metrics.total_contacts)} note="включая пилоты" />
      <Stat label="Общие затраты" value={number(run.metrics.total_cost, 2)} note="включая пилоты · у.е." />
      <Stat label="Чистый эффект ARPU" value={number(run.metrics.net_arpu_gain, 2)} note="оценка mock-модели · у.е." />
    </div>}
    {run.campaigns.length ? <DataTable label="Финальные кампании"
      headers={['Кампания', 'Целевой тариф', 'Канал', 'Сегмент', 'Охват', 'Затраты, у.е.', 'Gross lift, у.е.', 'Лимиты']}
      rows={run.campaigns.map((campaign) => [
        <strong>{campaign.campaign_name}</strong>, campaign.target_tariff, <span className="tag">{campaign.channel}</span>,
        <SegmentDetails filters={Object.fromEntries(Object.entries(campaign).filter(([key]) => key.startsWith('filter_')))} />,
        number(campaign.n_contacts), number(campaign.cost, 2), number(campaign.gross_lift, 2),
        [campaign.capped_at_campaign_limit && 'размер', campaign.capped_at_reach_budget && 'контакты',
          campaign.capped_at_money_budget && 'бюджет'].filter(Boolean).join(', ') || '—',
      ])} /> : <Notice>Backend завершил запуск без финальных кампаний.</Notice>}
  </>;
}
