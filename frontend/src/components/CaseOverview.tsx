import { useState } from 'react';
import type { CaseSummary } from '../types/api';
import { DataTable, number, Stat } from './Ui';

const segmentLabels: Record<string, string> = {
  arpu_segment: 'ARPU', data_segment: 'Интернет', call_segment: 'Звонки', current_tariff: 'Текущий тариф',
};

export function CaseOverview({ data }: { data: CaseSummary }) {
  const [segment, setSegment] = useState('arpu_segment');
  const limits = data.constraints;
  const selected = data.segments[segment] ? segment : Object.keys(data.segments)[0];
  return <>
    <div className="stats">
      <Stat label="Аудитория" value={number(data.customers)} note="клиентов в кейсе" />
      <Stat label="Средний ARPU" value={number(data.mean_predicted_arpu, 2)} note="прогноз · у.е." />
      <Stat label="Бюджет" value={number(limits.budget)} note="общий лимит · у.е." />
      <Stat label="Лимит контактов" value={number(limits.max_contacts)} note="включая пилоты" />
    </div>
    <div className="overview-grid">
      <section className="panel">
        <div className="panel-heading"><h3>Сегменты аудитории</h3>
          <label className="sr-only" htmlFor="segment-dimension">Разрез сегментов</label>
          <select id="segment-dimension" value={selected} onChange={(event) => setSegment(event.target.value)}>
            {Object.keys(data.segments).map((key) => <option key={key} value={key}>{segmentLabels[key] ?? key}</option>)}
          </select>
        </div>
        <div className="segment-list">{(data.segments[selected] ?? []).map((row, index) =>
          <div key={index} className="segment-row">
            <div><strong>{String(row[selected] ?? 'Не указан')}</strong><span>{number(row.customers)} клиентов</span></div>
            <progress value={row.customers} max={Math.max(data.customers, 1)} aria-label={String(row[selected] ?? 'Не указан')} />
            <small>Средний ARPU: {number(row.mean_predicted_arpu, 2)}</small>
          </div>)}</div>
      </section>
      <section className="panel">
        <div className="panel-heading"><h3>Лимиты кейса</h3><span className="eyebrow">CONSTRAINTS</span></div>
        <dl className="limits">
          <div><dt>Финальных кампаний</dt><dd>до {number(limits.max_campaigns)}</dd></div>
          <div><dt>Контактов на кампанию</dt><dd>до {number(limits.max_campaign_size)}</dd></div>
          <div><dt>Пилотных запусков</dt><dd>до {number(limits.max_pilots)}</dd></div>
          <div><dt>Размер пилота</dt><dd>{number(limits.min_pilot_size)}–{number(limits.max_pilot_size)}</dd></div>
          <div><dt>Время выполнения</dt><dd>{number(limits.max_runtime_seconds)} сек.</dd></div>
        </dl>
        <p className="muted">{limits.pilots_consume_budget_and_contacts ? 'Пилоты расходуют общий бюджет и лимит контактов.' : 'Лимиты пилотов учитываются отдельно.'}</p>
        <div className="channel-list">{Object.entries(data.channels).map(([name, channel]) =>
          <span className="tag" key={name}>{name} · {number(channel.cost_per_contact)} у.е./контакт</span>)}</div>
      </section>
    </div>
    <details className="panel tariff-panel">
      <summary>Тарифы <span className="count">{data.tariffs.length}</span></summary>
      <DataTable label="Тарифы" headers={['Тариф', 'Цена / мес., у.е.', 'Интернет, МБ', 'Минуты другим операторам', 'Минуты операторам + город', 'Описание']}
        rows={data.tariffs.map((tariff) => [tariff.tariff_plan_code, number(tariff.price_tariff, 2), number(tariff.Data_in_PKG),
          number(tariff.Min_another_operator_in_PKG), number(tariff.Min_another_operator_and_city_in_PKG), <span className="description">{tariff.description}</span>])} />
    </details>
  </>;
}
