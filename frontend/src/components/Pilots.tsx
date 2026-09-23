import type { Pilot } from '../types/api';
import { displayValue } from '../utils/labels';
import { DataTable, Notice, number, percent, SegmentDetails } from './Ui';

export function Pilots({ pilots, loading, links = {} }: {
  pilots: Pilot[];
  loading: boolean;
  links?: Record<string, string[]>;
}) {
  if (!pilots.length) return <Notice>{loading ? 'Ожидаем первые результаты пробных запусков…' : 'Пробных запусков пока нет.'}</Notice>;
  return <DataTable label="Пробные запуски"
    headers={['Пилот', 'Кандидат',
      { label: 'Наблюдаемый результат', help: 'Изменение дохода на клиента, наблюдаемое на канале пробного запуска.' },
      { label: 'Оценка (mu)', help: 'Оценка базового эффекта после пилота, до множителя канала. Значение получено от агента.' },
      { label: 'LCB', help: 'Нижняя граница оценки эффекта с учётом неопределённости. Положительное значение подтверждает гипотезу по критерию агента.' },
      'Канал', 'Клиентов', 'Стоимость, у.е.', 'Кампания', 'Вывод']}
    rows={pilots.map((pilot) => {
      const pilotId = 'P' + pilot.sequence;
      const campaigns = Object.entries(links).filter(([, ids]) => ids.includes(pilotId)).map(([id]) => id);
      const hasLcb = typeof pilot.lcb === 'number' && Number.isFinite(pilot.lcb);
      return [
        <strong>{pilotId}</strong>,
        <div className="candidate-cell">
          <span>{displayValue(pilot.target_tariff)}</span>
          <SegmentDetails filters={pilot.segment} />
          {pilot.candidate_id ? <details><summary>Идентификатор кандидата</summary><code>{pilot.candidate_id}</code></details> : <small>Кандидат не указан</small>}
          <details><summary>Гипотеза</summary><p>{pilot.hypothesis}</p></details>
        </div>,
        percent(pilot.observed_lift), percent(pilot.mu), percent(pilot.lcb),
        <span className="tag">{displayValue(pilot.channel)}</span>, number(pilot.pilot_size), number(pilot.cost, 2),
        campaigns.join(', ') || '—',
        hasLcb ? <span className={'status ' + (pilot.lcb! > 0 ? 'status-completed' : 'status-error')}>
          {pilot.lcb! > 0 ? 'Подтверждено' : 'Отклонено'}</span> : '—',
      ];
    })} />;
}
