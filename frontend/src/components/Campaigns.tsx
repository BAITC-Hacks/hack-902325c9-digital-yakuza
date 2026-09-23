import type { AgentRun } from '../types/api';
import { campaignLabel, displayValue } from '../utils/labels';
import { DataTable, Notice, number, SegmentDetails, Stat } from './Ui';

export function Campaigns({ run }: { run: AgentRun | null }) {
  if (!run || run.status !== 'completed') return <Notice>{run?.status === 'running'
    ? 'Агент проводит пробные запуски. Финальные кампании появятся после завершения.'
    : 'Завершите запуск агента, чтобы получить финальные кампании.'}</Notice>;
  return <>
    {run.metrics && <div className="stats">
      <Stat label="Финальные кампании" value={number(run.campaigns.length)} />
      <Stat label="Всего контактов" help="Общее число обращений к клиентам в пробных и финальных кампаниях. Это не обязательно число уникальных клиентов." value={number(run.metrics.total_contacts)}  />
      <Stat label="Общие затраты" help="Стоимость обращений к клиентам, включая пробные запуски." value={number(run.metrics.total_cost, 2)} note="у.е." />
      <Stat label="Чистый эффект" help="Оценка прироста дохода за вычетом затрат на коммуникацию. Рассчитана моделью; фактический результат может отличаться." value={number(run.metrics.net_arpu_gain, 2)} note="у.е." />
    </div>}
    {run.campaigns.length ? <DataTable label="Финальные кампании"
      headers={['Кампания', 'Целевой тариф', 'Канал', 'Сегмент', 'Охват', 'Затраты, у.е.', { label: 'Прирост дохода, у.е.', help: 'Оценка дополнительного дохода от кампании до вычета затрат на коммуникацию.' }, { label: 'Ограничения охвата', help: 'Показывает, какие лимиты сократили охват кампании: размер одной кампании, общее число контактов или бюджет.' }]}
      rows={run.campaigns.map((campaign) => [
        <strong>{campaignLabel(campaign.campaign_name)}</strong>, displayValue(campaign.target_tariff), <span className="tag">{displayValue(campaign.channel)}</span>,
        <SegmentDetails filters={Object.fromEntries(Object.entries(campaign).filter(([key]) => key.startsWith('filter_')))} />,
        number(campaign.n_contacts), number(campaign.cost, 2), number(campaign.gross_lift, 2),
        [campaign.capped_at_campaign_limit && 'размер', campaign.capped_at_reach_budget && 'контакты',
          campaign.capped_at_money_budget && 'бюджет'].filter(Boolean).join(', ') || '—',
      ])} /> : <Notice>Агент завершил запуск без финальных кампаний.</Notice>}
  </>;
}
